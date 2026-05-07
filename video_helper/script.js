const video = document.getElementById('video');
const fileInput = document.getElementById('fileInput');
const fpsReadout = document.getElementById('fpsReadout');
const timeReadout = document.getElementById('timeReadout');
const frameReadout = document.getElementById('frameReadout');
const stepMode = document.getElementById('stepMode');
const stepSizeInput = document.getElementById('stepSize');
const stepBackBtn = document.getElementById('stepBack');
const stepForwardBtn = document.getElementById('stepForward');
const saveTimestampBtn = document.getElementById('saveTimestampBtn');
const clearTimestampsBtn = document.getElementById('clearTimestampsBtn');
const exportJsonBtn = document.getElementById('exportJsonBtn');
const timestampsList = document.getElementById('timestampsList');
const gestureJsonInput = document.getElementById('gestureJsonInput');
const gestureIndicator = document.getElementById('gestureIndicator');
const indicatorValue = document.getElementById('indicatorValue');
const indicatorInfo = document.getElementById('indicatorInfo');
const gestureJsonInput2 = document.getElementById('gestureJsonInput_code');
const gestureIndicator2 = document.getElementById('gestureIndicator_code');
const indicatorValue2 = document.getElementById('indicatorValue_code');
const indicatorInfo2 = document.getElementById('indicatorInfo_code');


let detectedFPS = 0;
let lastFrameTime = null;
let fpsSamples = [];
let fpsLocked = false;
let savedTimestamps = [];  // Array to store saved timestamps
let gestureData = [];  // Array to store loaded gesture data


fileInput.addEventListener('change', e => {
const file = e.target.files[0];
if (file) {
video.src = URL.createObjectURL(file);
video.load();
startBackgroundFPSDetection();
}
});


function startBackgroundFPSDetection() {
detectedFPS = 0;
fpsSamples = [];
lastFrameTime = null;
fpsLocked = false;
fpsReadout.textContent = 'Detecting…';


// Play silently in background for FPS detection
video.muted = true;
video.play().catch(() => {});
}


function lockFPS() {
if (fpsSamples.length < 10) return;
const avg = fpsSamples.reduce((a, b) => a + b, 0) / fpsSamples.length;
detectedFPS = Math.round(avg * 1000) / 1000;
fpsLocked = true;
fpsReadout.textContent = detectedFPS.toFixed(3);
video.pause();
}


function updateReadout(time) {
timeReadout.textContent = time.toFixed(3);
if (detectedFPS > 0) {
frameReadout.textContent = Math.round(time * detectedFPS);
} else {
frameReadout.textContent = '–';
}
}


function step(direction) {
const stepSize = parseFloat(stepSizeInput.value);
if (isNaN(stepSize) || stepSize <= 0) return;


if (stepMode.value === 'time') {
video.currentTime = Math.max(0, video.currentTime + direction * stepSize);
} else if (stepMode.value === 'frame' && detectedFPS > 0) {
const delta = direction * (stepSize / detectedFPS);
video.currentTime = Math.max(0, video.currentTime + delta);
}
}


stepBackBtn.onclick = () => step(-1);
stepForwardBtn.onclick = () => step(1);


video.addEventListener('timeupdate', () => {
updateReadout(video.currentTime);
handler1.update(video.currentTime);
handler2.update(video.currentTime);
});


// One-time FPS detection during background playback
if ('requestVideoFrameCallback' in HTMLVideoElement.prototype) {
const frameCallback = (now, metadata) => {
if (!fpsLocked) {
if (lastFrameTime !== null) {
const delta = metadata.mediaTime - lastFrameTime;
if (delta > 0 && delta < 0.1) {
fpsSamples.push(1 / delta);
if (fpsSamples.length > 30) {
lockFPS();
return;
}
}
}
lastFrameTime = metadata.mediaTime;
video.requestVideoFrameCallback(frameCallback);
}
};


video.requestVideoFrameCallback(frameCallback);
} else {
fpsReadout.textContent = 'Unsupported';
}


// Timestamp saving functionality
function saveTimestamp() {
const currentTime = video.currentTime;
const currentFrame = detectedFPS > 0 ? Math.round(currentTime * detectedFPS) : 0;

// If this is the first timestamp, add the 0:0 starting point
if (savedTimestamps.length === 0) {
savedTimestamps.push({
time: 0,
frame: 0,
id: -1  // Special ID for the initial 0:0 marker
});
}

const timestamp = {
time: parseFloat(currentTime.toFixed(3)),
frame: currentFrame,
id: Date.now()
};

savedTimestamps.push(timestamp);
updateTimestampsList();
}


function removeTimestamp(id) {
// Don't allow removing the initial 0:0 marker
if (id === -1) {
alert('Cannot remove the initial 0:0 marker.');
return;
}

savedTimestamps = savedTimestamps.filter(ts => ts.id !== id);

// If no timestamps left except the initial marker, remove it too
if (savedTimestamps.length === 1 && savedTimestamps[0].id === -1) {
savedTimestamps = [];
}

updateTimestampsList();
}


function goToClip(index) {
if (index >= 0 && index < savedTimestamps.length) {
video.currentTime = savedTimestamps[index].time;
video.play();
}
}


function updateTimestampsList() {
if (savedTimestamps.length === 0) {
timestampsList.innerHTML = '<p class="no-timestamps">No timestamps saved yet. Press S or click "Save Timestamp" to mark cuts.</p>';
return;
}

let html = '';
for (let i = 0; i < savedTimestamps.length; i++) {
const ts = savedTimestamps[i];
const nextTs = savedTimestamps[i + 1];
const clipInfo = nextTs ? `Clip ${i}: ${ts.time}s - ${nextTs.time}s` : `Start: ${ts.time}s`;
const duration = nextTs ? parseFloat((nextTs.time - ts.time).toFixed(2)) : '–';
const isInitial = ts.id === -1;

html += `
<div class="timestamp-item" onclick="goToClip(${i})" style="cursor: pointer;">
<div class="time-info">
<strong>${isInitial ? 'START' : '#' + i}</strong> | Frame: ${ts.frame} | Time: ${ts.time}s | Duration: ${duration}s
<div class="cut-info">${clipInfo}</div>
</div>
${isInitial ? '' : `<button onclick="event.stopPropagation(); removeTimestamp(${ts.id})">Remove</button>`}
</div>
`;
}

timestampsList.innerHTML = html;
}


function exportAsJSON() {
if (savedTimestamps.length < 2) {
alert('Please save at least 1 timestamp to create clips. (Clip 0 from 0:0 to your first timestamp will be included)');
return;
}

const videoDuration = video.duration || 0;
const clips = [];

for (let i = 0; i < savedTimestamps.length - 1; i++) {
const startTs = savedTimestamps[i];
const endTs = savedTimestamps[i + 1];

clips.push({
clip_id: i,
start_frame: startTs.frame,
end_frame: endTs.frame,
start_time: startTs.time,
end_time: endTs.time,
duration: parseFloat((endTs.time - startTs.time).toFixed(2))
});
}

// Add final clip from last timestamp to end of video
const lastTs = savedTimestamps[savedTimestamps.length - 1];
const finalEndFrame = detectedFPS > 0 ? Math.round(videoDuration * detectedFPS) : 0;
clips.push({
clip_id: savedTimestamps.length - 1,
start_frame: lastTs.frame,
end_frame: finalEndFrame,
start_time: lastTs.time,
end_time: parseFloat(videoDuration.toFixed(3)),
duration: parseFloat((videoDuration - lastTs.time).toFixed(2))
});

const exportData = {
fps: detectedFPS,
total_clips: clips.length,
clips: clips,
video_duration: parseFloat(videoDuration.toFixed(3)),
generated_at: new Date().toISOString()
};

const jsonString = JSON.stringify(exportData, null, 2);
const blob = new Blob([jsonString], { type: 'application/json' });
const url = URL.createObjectURL(blob);
const a = document.createElement('a');
a.href = url;
a.download = `video_clips_${Date.now()}.json`;
a.click();
URL.revokeObjectURL(url);
}


function clearAllTimestamps() {
if (savedTimestamps.length > 0 && confirm('Are you sure you want to clear all timestamps?')) {
savedTimestamps = [];
updateTimestampsList();
}
}


// Event listeners for buttons
saveTimestampBtn.onclick = saveTimestamp;
clearTimestampsBtn.onclick = clearAllTimestamps;
exportJsonBtn.onclick = exportAsJSON;

// Keyboard shortcut: Press S to save timestamp
document.addEventListener('keydown', (e) => {
if (e.key.toLowerCase() === 's' && !e.ctrlKey && !e.metaKey) {
e.preventDefault();
saveTimestamp();
}
});


function createGestureHandler({
    inputEl,
    indicatorEl,
    valueEl,
    infoEl
}) {
    let gestureData = [];

    inputEl.addEventListener('change', (e) => {
        const file = e.target.files[0];
        if (file) {
            const reader = new FileReader();

            reader.onload = (event) => {
                try {
                    const data = JSON.parse(event.target.result);

                    if (Array.isArray(data)) {
                        gestureData = data;
                    } else if (data.clips && Array.isArray(data.clips)) {
                        gestureData = data.clips;
                    } else if (data.gestures && Array.isArray(data.gestures)) {
                        gestureData = data.gestures;
                    } else {
                        throw new Error('Invalid JSON format');
                    }

                    console.log(`Loaded ${gestureData.length} gestures`);
                    indicatorEl.classList.remove('hidden');
                    updateIndicator(video.currentTime);
                } catch (error) {
                    alert('Error loading gesture JSON: ' + error.message);
                    gestureData = [];
                }
            };

            reader.readAsText(file);
        }
    });

    function updateIndicator(currentTime) {
        if (gestureData.length === 0) return;

        let matchingGesture = null;

        for (const gesture of gestureData) {
            const startTime = gesture.start_time || gesture.start;
            const endTime = gesture.end_time || gesture.end;

            if (currentTime >= startTime && currentTime <= endTime) {
                matchingGesture = gesture;
                break;
            }
        }

        if (matchingGesture) {
            valueEl.textContent = 'YES';
            indicatorEl.classList.remove('active-no');
            indicatorEl.classList.add('active-yes');

            const gestureName = matchingGesture.gesture || matchingGesture.clip_id || 'Gesture';
            const duration =
                (matchingGesture.end_time || matchingGesture.end) -
                (matchingGesture.start_time || matchingGesture.start);

            infoEl.textContent = `${gestureName} (${duration.toFixed(2)}s)`;
        } else {
            valueEl.textContent = 'NO';
            indicatorEl.classList.remove('active-yes');
            indicatorEl.classList.add('active-no');
            infoEl.textContent = 'No gesture at current timestamp';
        }
    }

    return {
        update: updateIndicator
    };
}

const handler1 = createGestureHandler({
    inputEl: gestureJsonInput,
    indicatorEl: gestureIndicator,
    valueEl: indicatorValue,
    infoEl: indicatorInfo
});

const handler2 = createGestureHandler({
    inputEl: gestureJsonInput2,
    indicatorEl: gestureIndicator2,
    valueEl: indicatorValue2,
    infoEl: indicatorInfo2
});