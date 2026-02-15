let video;
let poseNet;
let poses = [];
let particles = [];
let contactPoints = [];
let contactArea = 0;
let trackedPoses = new Map();
let nextPoseId = 0;

const CONTACT_THRESHOLD = 32;
const MIN_POSE_SCORE = 0.05;
const MIN_PART_SCORE = 0.02;
const BODY_SAMPLE_STEPS = 12; // Increased from 8 for smoother fill
const TORSO_STEPS = 10;       // Increased from 6 for smoother fill
const CONTACT_SPATIAL_CELL = 35; // Decreased from 45 for tighter buckets
const CONTACT_MERGE_DISTANCE = 4; // Decreased from 8 to allow denser points
const MAX_CONTACT_POINTS = 1500;  // Increased from 900
const SMOOTHING_FACTOR = 0.15;    // Reduced for smoother movement
const MATCH_DISTANCE = 150;       // Increased from 120 for better close-contact tracking
const PARTICLE_LIMIT = 10000;
const PARTICLE_MIN_SIZE = 2.0;
const PARTICLE_MAX_SIZE = 4.0;
const PARTICLE_JITTER = 0.45;     // Slightly increased for more organic motion
const PARTICLE_COLORS = [
    [255, 0, 0]
];

const SKELETON_CONNECTIONS = [
    ['nose', 'leftEye'], ['nose', 'rightEye'],
    ['leftEye', 'leftEar'], ['rightEye', 'rightEar'],
    ['leftShoulder', 'rightShoulder'],
    ['leftShoulder', 'leftElbow'], ['leftElbow', 'leftWrist'],
    ['rightShoulder', 'rightElbow'], ['rightElbow', 'rightWrist'],
    ['leftShoulder', 'leftHip'], ['rightShoulder', 'rightHip'],
    ['leftHip', 'rightHip'],
    ['leftHip', 'leftKnee'], ['leftKnee', 'leftAnkle'],
    ['rightHip', 'rightKnee'], ['rightKnee', 'rightAnkle']
];

const viewTransform = {
    scale: 1,
    offsetX: 0,
    offsetY: 0,
    drawWidth: 0,
    drawHeight: 0,
    sourceWidth: 640,
    sourceHeight: 480
};

function setup() {
    createCanvas(windowWidth, windowHeight);

    video = createCapture(VIDEO, () => {
        if (video.elt && video.elt.videoWidth) {
            viewTransform.sourceWidth = video.elt.videoWidth;
            viewTransform.sourceHeight = video.elt.videoHeight;
        }
    });
    video.size(640, 480);
    video.hide();
    video.elt.setAttribute('playsinline', '');
    video.style('transform', 'scaleX(1)');

    poseNet = ml5.poseNet(video, {
        detectionType: 'multiple',
        flipHorizontal: false,
        maxPoseDetections: 2,
        scoreThreshold: MIN_PART_SCORE,
        minConfidence: MIN_PART_SCORE,
        nmsRadius: 15 // Slightly tighter than 10/25 to avoid merging but allow close proximity
    }, modelReady);

    poseNet.on('pose', gotPoses);
}

function windowResized() {
    resizeCanvas(windowWidth, windowHeight);
}

function mousePressed() {
    const pg = createGraphics(width, height);
    pg.clear();
    pg.noStroke();
    pg.fill(0, 0, 0); // Pure black particles for export

    // Snapshot particles to avoid mutations during draw/update cycles
    const snapshot = particles.map(p => ({
        x: p.x,
        y: p.y,
        life: p.life,
        maxLife: p.maxLife,
        size: p.size
    }));

    for (let i = 0; i < snapshot.length; i++) {
        const p = snapshot[i];
        if (!p) continue;
        const lifeRatio = p.maxLife ? max(p.life / p.maxLife, 0) : 0;
        const size = p.size * lifeRatio;

        if (size > 0.05) {
            const projected = projectPoint(p);
            if (!projected || !isFinite(projected.x) || !isFinite(projected.y)) continue;
            pg.ellipse(projected.x, projected.y, size * viewTransform.scale, size * viewTransform.scale);
        }
    }

    // Generate filename with timestamp
    const timestamp = year() + nf(month(), 2) + nf(day(), 2) + '_' + nf(hour(), 2) + nf(minute(), 2) + nf(second(), 2);
    save(pg, 'contact_snapshot_' + timestamp + '.png');
    pg.remove();
}

function modelReady() {
    console.log('PoseNet model loaded');
}

function gotPoses(results) {
    poses = smoothAndFormatPoses(results);
    calculateContactArea();
}

function smoothAndFormatPoses(detections) {
    const formatted = [];
    const usedIds = new Set();
    
    // Enforce tracking of up to 2 distinct bodies
    // We sort by confidence to prioritize the best matches first
    const sortedDetections = detections.sort((a, b) => b.pose.score - a.pose.score);

    for (let i = 0; i < sortedDetections.length; i++) {
        const rawPose = sortedDetections[i].pose;
        if (!rawPose || !rawPose.keypoints) continue;

        // Simple duplicate check based on center distance
        // If this new pose is very close to an already accepted pose, skip it
        const rawCenter = getKeypointCenter(rawPose.keypoints);
        let isDuplicate = false;
        for (let j = 0; j < formatted.length; j++) {
            const existing = formatted[j].pose;
            const existingCenter = getKeypointCenter(existing.keypoints);
            if (dist(rawCenter.x, rawCenter.y, existingCenter.x, existingCenter.y) < 50) {
                isDuplicate = true;
                break;
            }
        }
        if (isDuplicate) continue;

        const id = matchTrackedCenter(rawPose, usedIds);
        const previous = trackedPoses.get(id);
        const smoothedPose = applySmoothingToPose(rawPose, previous);
        const skeleton = buildSkeletonFromPose(smoothedPose);
        const keypointMap = mapKeypoints(smoothedPose.keypoints);
        const center = getKeypointCenter(smoothedPose.keypoints);

        trackedPoses.set(id, {
            keypoints: keypointMap,
            center,
            lastSeen: frameCount
        });
        usedIds.add(id);

        smoothedPose.skeleton = skeleton;

        formatted.push({
            pose: smoothedPose,
            skeleton,
            id
        });
        
        // Only process up to 2 bodies to prevent ghosting/merging artifacts
        if (formatted.length >= 2) break;
    }

    for (const [id, track] of trackedPoses.entries()) {
        if (track.lastSeen < frameCount - 30) { // Faster timeout for stale IDs
            trackedPoses.delete(id);
        }
    }

    return formatted;
}

function matchTrackedCenter(rawPose, usedIds) {
    const center = getKeypointCenter(rawPose.keypoints);
    let matchId = null;
    let bestDist = MATCH_DISTANCE;

    for (const [id, track] of trackedPoses.entries()) {
        if (usedIds.has(id)) continue;
        if (!track.center) continue;
        const d = dist(center.x, center.y, track.center.x, track.center.y);
        if (d < bestDist) {
            bestDist = d;
            matchId = id;
        }
    }

    if (matchId === null) {
        matchId = nextPoseId++;
    }

    return matchId;
}

function applySmoothingToPose(rawPose, previous) {
    const smoothedKeypoints = rawPose.keypoints.map(kp => {
        const prev = previous?.keypoints?.[kp.part];
        const smoothedPosition = prev
            ? {
                x: lerp(prev.x, kp.position.x, SMOOTHING_FACTOR),
                y: lerp(prev.y, kp.position.y, SMOOTHING_FACTOR)
            }
            : { x: kp.position.x, y: kp.position.y };

    return {
            ...kp,
            position: smoothedPosition
        };
    });

    return {
        ...rawPose,
        keypoints: smoothedKeypoints
    };
}

function mapKeypoints(keypoints) {
    const map = {};
    for (let i = 0; i < keypoints.length; i++) {
        const kp = keypoints[i];
        map[kp.part] = {
            x: kp.position.x,
            y: kp.position.y,
            score: kp.score
        };
    }
    return map;
}

function getKeypointCenter(keypoints) {
    let sumX = 0;
    let sumY = 0;
    let count = 0;
    for (let i = 0; i < keypoints.length; i++) {
        const kp = keypoints[i];
        if (kp.score > MIN_PART_SCORE) {
            sumX += kp.position.x;
            sumY += kp.position.y;
            count++;
        }
    }
    if (count === 0) {
        return { x: 0, y: 0 };
    }
    return { x: sumX / count, y: sumY / count };
}

function buildSkeletonFromPose(pose) {
    const keypointMap = mapKeypoints(pose.keypoints);
    const skeleton = [];
    for (let i = 0; i < SKELETON_CONNECTIONS.length; i++) {
        const [partA, partB] = SKELETON_CONNECTIONS[i];
        const a = keypointMap[partA];
        const b = keypointMap[partB];
        if (!a || !b) continue;
        if (a.score > MIN_PART_SCORE && b.score > MIN_PART_SCORE) {
            skeleton.push([
                { score: a.score, position: { x: a.x, y: a.y } },
                { score: b.score, position: { x: b.x, y: b.y } }
            ]);
        }
    }
    return skeleton;
}

function shuffleInPlace(array) {
    for (let i = array.length - 1; i > 0; i--) {
        const j = Math.floor(Math.random() * (i + 1));
        [array[i], array[j]] = [array[j], array[i]];
    }
}

function isDistinctContact(points, candidate, minDistance) {
    // More optimized check for dense clouds
    // Only check recent points to keep it O(N) approx
    const lookback = Math.min(points.length, 40); 
    for (let i = points.length - 1; i >= points.length - lookback; i--) {
        const existing = points[i];
        const d = Math.abs(candidate.x - existing.x) + Math.abs(candidate.y - existing.y); // Manhattan dist approx for speed
        if (d < minDistance) {
            return false;
        }
    }
    return true;
}

function draw() {
    background(0);

    if (!video || !(video.elt && video.elt.videoWidth)) {
        return;
    }

    updateViewTransform();
    drawVideo();
    drawAtmosphere();
    drawPoses();
    updateParticles();
    drawContactInfo();
}

function drawContactInfo() {
    push();
    fill(255);
    noStroke();
    textSize(24);
    textAlign(LEFT, TOP);
    text(`Contact Value: ${Math.round(contactArea)}`, 20, 20);
    pop();
}

function updateViewTransform() {
    const sourceWidth = video.elt.videoWidth || viewTransform.sourceWidth;
    const sourceHeight = video.elt.videoHeight || viewTransform.sourceHeight;
    viewTransform.sourceWidth = sourceWidth;
    viewTransform.sourceHeight = sourceHeight;

    const scale = min(width / sourceWidth, height / sourceHeight);
    viewTransform.scale = scale;
    viewTransform.drawWidth = sourceWidth * scale;
    viewTransform.drawHeight = sourceHeight * scale;
    viewTransform.offsetX = (width - viewTransform.drawWidth) / 2;
    viewTransform.offsetY = (height - viewTransform.drawHeight) / 2;
}

function drawVideo() {
    push();
    translate(viewTransform.offsetX + viewTransform.drawWidth, viewTransform.offsetY);
    scale(-viewTransform.scale, viewTransform.scale);
    image(video, 0, 0, viewTransform.sourceWidth, viewTransform.sourceHeight);
    pop();
}

function drawAtmosphere() {
    noStroke();
    fill(0, 170);
    rect(0, 0, width, height);
}

function drawPoses() {
    if (poses.length < 2) return; // Hide lines if only one person is tracked

    for (let i = 0; i < poses.length; i++) {
        const pose = poses[i].pose;
        const skeleton = poses[i].skeleton || [];

        stroke(255);
        strokeWeight(3);

        for (let j = 0; j < skeleton.length; j++) {
            const partA = skeleton[j][0];
            const partB = skeleton[j][1];
            const a = projectPoint(partA.position);
            const b = projectPoint(partB.position);
            line(a.x, a.y, b.x, b.y);
        }

        noStroke();
        fill(255);
        for (let j = 0; j < pose.keypoints.length; j++) {
            const keypoint = pose.keypoints[j];
            if (keypoint.score > MIN_PART_SCORE) {
                const projected = projectPoint(keypoint.position);
                ellipse(projected.x, projected.y, 8, 8);
            }
        }
    }
}

function projectPoint(point) {
    const x = viewTransform.offsetX + (viewTransform.sourceWidth - point.x) * viewTransform.scale;
    const y = viewTransform.offsetY + point.y * viewTransform.scale;
    return { x, y };
}

function calculateContactArea() {
    contactArea = 0;
    contactPoints = [];

    if (poses.length < 2) {
        particles.length = 0;
        return;
    }

    // We assume pose[0] and pose[1] are the two distinct people due to our tracking logic
    const poseA = poses[0].pose;
    const poseB = poses[1].pose;

    const samplesA = buildBodyField(poseA);
    const samplesB = buildBodyField(poseB);
    if (!samplesA.length || !samplesB.length) {
        particles.length = 0;
        return;
    }

    const hashB = buildSpatialHash(samplesB, CONTACT_SPATIAL_CELL);

    // Randomize sampling order to avoid line-scan artifacts
    shuffleInPlace(samplesA);

    for (let i = 0; i < samplesA.length; i++) {
        const sa = samplesA[i];
        const neighbors = getNeighborPoints(hashB, sa, CONTACT_SPATIAL_CELL);
        for (let j = 0; j < neighbors.length; j++) {
            const sb = neighbors[j];
            const d = dist(sa.x, sa.y, sb.x, sb.y);
            if (d < CONTACT_THRESHOLD) {
                // Add random jitter to the spawn point to break linearity
                const mid = {
                    x: (sa.x + sb.x) / 2 + random(-2, 2),
                    y: (sa.y + sb.y) / 2 + random(-2, 2),
                    weight: (sa.weight + sb.weight) / 2
                };
                if (isDistinctContact(contactPoints, mid, CONTACT_MERGE_DISTANCE)) {
                    contactPoints.push(mid);
                }
            }
        }
    }

    if (contactPoints.length > MAX_CONTACT_POINTS) {
        shuffleInPlace(contactPoints);
        contactPoints.length = MAX_CONTACT_POINTS;
    }

    contactArea = contactPoints.reduce((sum, pt) => sum + pt.weight, 0);

    if (contactPoints.length) {
        emitContactParticles(contactPoints);
    } else {
        particles.length = 0;
    }
}

function buildBodyField(pose) {
    const samples = [];
    const skeleton = pose.skeleton || [];

    for (let i = 0; i < skeleton.length; i++) {
        const partA = skeleton[i][0];
        const partB = skeleton[i][1];
        if (partA.score < MIN_PART_SCORE || partB.score < MIN_PART_SCORE) continue;
        for (let step = 0; step <= BODY_SAMPLE_STEPS; step++) {
            // Add slight noise to t to avoid perfect grid sampling along lines
            const t = constrain((step / BODY_SAMPLE_STEPS) + random(-0.05, 0.05), 0, 1);
            samples.push({
                x: lerp(partA.position.x, partB.position.x, t),
                y: lerp(partA.position.y, partB.position.y, t),
                weight: 1
            });
        }
    }

    return samples.concat(sampleTorso(pose));
}

function sampleTorso(pose) {
    const leftShoulder = getKeypoint(pose, 'leftShoulder');
    const rightShoulder = getKeypoint(pose, 'rightShoulder');
    const leftHip = getKeypoint(pose, 'leftHip');
    const rightHip = getKeypoint(pose, 'rightHip');

    if (!leftShoulder || !rightShoulder || !leftHip || !rightHip) {
        return [];
    }

    if (
        leftShoulder.score < MIN_PART_SCORE ||
        rightShoulder.score < MIN_PART_SCORE ||
        leftHip.score < MIN_PART_SCORE ||
        rightHip.score < MIN_PART_SCORE
    ) {
        return [];
    }

    const samples = [];
    for (let h = 0; h <= TORSO_STEPS; h++) {
        const t = h / TORSO_STEPS;
        const top = lerpPoint(leftShoulder.position, rightShoulder.position, t);
        const bottom = lerpPoint(leftHip.position, rightHip.position, t);
        for (let v = 0; v <= TORSO_STEPS; v++) {
            // Add 2D jitter to break grid patterns
            const vt = constrain((v / TORSO_STEPS) + random(-0.05, 0.05), 0, 1);
            const sample = lerpPoint(top, bottom, vt);
            samples.push({
                x: sample.x + random(-2, 2),
                y: sample.y + random(-2, 2),
                weight: 1.2
            });
        }
    }
    return samples;
}

function lerpPoint(p1, p2, t) {
    return {
        x: lerp(p1.x, p2.x, t),
        y: lerp(p1.y, p2.y, t)
    };
}

function getKeypoint(pose, partName) {
    return pose.keypoints.find(k => k.part === partName);
}

function buildSpatialHash(points, cellSize) {
    const hash = new Map();
    for (let i = 0; i < points.length; i++) {
        const pt = points[i];
        const key = `${Math.floor(pt.x / cellSize)}-${Math.floor(pt.y / cellSize)}`;
        if (!hash.has(key)) {
            hash.set(key, []);
        }
        hash.get(key).push(pt);
    }
    return hash;
}

function getNeighborPoints(hash, point, cellSize) {
    const neighbors = [];
    const cx = Math.floor(point.x / cellSize);
    const cy = Math.floor(point.y / cellSize);
    for (let dx = -1; dx <= 1; dx++) {
        for (let dy = -1; dy <= 1; dy++) {
            const key = `${cx + dx}-${cy + dy}`;
            if (hash.has(key)) {
                neighbors.push(...hash.get(key));
            }
        }
    }
    return neighbors;
}

function emitContactParticles(points) {
    if (!points.length) return;

    // High density emission for fluid look
    const density = constrain(points.length / 1.5, 20, 150);

    for (let i = 0; i < points.length; i++) {
        const pt = points[i];
        // Variance in spawn count per point for organic feel
        const spawns = Math.round(random(density * 0.8, density * 1.2));
        for (let j = 0; j < spawns; j++) {
            if (particles.length >= PARTICLE_LIMIT) {
                particles.splice(0, particles.length - PARTICLE_LIMIT + 1);
            }
            particles.push(createParticle(pt));
        }
    }
}

function createParticle(origin) {
    const color = random(PARTICLE_COLORS);
    const maxLife = random(25, 60);
    // Spread spawn radius slightly to fill gaps between contact points
    const spread = 3.5;
    return {
        x: origin.x + random(-spread, spread),
        y: origin.y + random(-spread, spread),
        vx: random(-0.15, 0.15),
        vy: random(-0.15, 0.15),
        size: random(PARTICLE_MIN_SIZE, PARTICLE_MAX_SIZE),
        life: maxLife,
        maxLife,
        color
    };
}

function updateParticles() {
    for (let i = particles.length - 1; i >= 0; i--) {
        const p = particles[i];
        p.x += p.vx + random(-PARTICLE_JITTER, PARTICLE_JITTER);
        p.y += p.vy + random(-PARTICLE_JITTER, PARTICLE_JITTER);
        p.life -= 1;

        const lifeRatio = max(p.life / p.maxLife, 0);
        const projected = projectPoint(p);
        const size = p.size * lifeRatio;

        if (size <= 0.05) {
            particles.splice(i, 1);
            continue;
        }

        noStroke();
        fill(p.color[0], p.color[1], p.color[2]);
        ellipse(projected.x, projected.y, size * viewTransform.scale, size * viewTransform.scale);

        if (p.life <= 0) {
            particles.splice(i, 1);
        }
    }
}
