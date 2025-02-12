let scene, camera, renderer, controls;
let dataPoints = [], accumulatorPoints = [], topPositions = [];
let gui;

const settings = {
    showDataPoints: true,
    showAccumulator: true,
    showTopPositions: true,
    accumulatorOpacity: 0.3,
    minAccValue: 0,
    maxAccValue: 1
};

init();
animate();

function init() {
    // Scene setup
    scene = new THREE.Scene();
    scene.background = new THREE.Color(0x111111);
    
    // Camera setup
    camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 1000);
    camera.position.set(50, 50, 50);
    
    // Renderer setup
    renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setSize(window.innerWidth, window.innerHeight);
    document.body.appendChild(renderer.domElement);
    
    // Controls
    controls = new THREE.OrbitControls(camera, renderer.domElement);
    
    // Lighting
    const ambientLight = new THREE.AmbientLight(0xffffff, 0.5);
    scene.add(ambientLight);
    const directionalLight = new THREE.DirectionalLight(0xffffff, 0.5);
    directionalLight.position.set(10, 10, 10);
    scene.add(directionalLight);
    
    // Load data
    loadData();
    
    // GUI
    setupGUI();
    
    // Handle window resize
    window.addEventListener('resize', onWindowResize, false);
}

function loadData() {
    fetch('visibility_data.json')
        .then(response => response.json())
        .then(data => {
            createDataPoints(data.dataPoints);
            createAccumulatorPoints(data.accumulatorPoints);
            createTopPositions(data.topPositions);
        });
}

function createDataPoints(points) {
    const geometry = new THREE.SphereGeometry(0.3);
    points.forEach(point => {
        const material = new THREE.MeshPhongMaterial({ color: point.color });
        const sphere = new THREE.Mesh(geometry, material);
        sphere.position.set(...point.position);
        dataPoints.push(sphere);
        scene.add(sphere);
    });
}

function createAccumulatorPoints(points) {
    const geometry = new THREE.BoxGeometry(1, 1, 1);
    const maxValue = Math.max(...points.map(p => p.value));
    
    points.forEach(point => {
        const intensity = point.value / maxValue;
        const material = new THREE.MeshPhongMaterial({
            color: new THREE.Color().setHSL(0.6, 1, intensity),
            transparent: true,
            opacity: settings.accumulatorOpacity
        });
        const cube = new THREE.Mesh(geometry, material);
        cube.position.set(...point.position);
        cube.userData.value = point.value;
        accumulatorPoints.push(cube);
        scene.add(cube);
    });
}

function createTopPositions(points) {
    const geometry = new THREE.SphereGeometry(0.5);
    points.forEach(point => {
        const material = new THREE.MeshPhongMaterial({ color: 0xff0000 });
        const sphere = new THREE.Mesh(geometry, material);
        sphere.position.set(...point.position);
        topPositions.push(sphere);
        scene.add(sphere);
    });
}

function setupGUI() {
    gui = new dat.GUI();
    
    gui.add(settings, 'showDataPoints').onChange(value => {
        dataPoints.forEach(point => point.visible = value);
    });
    
    gui.add(settings, 'showAccumulator').onChange(value => {
        accumulatorPoints.forEach(point => point.visible = value);
    });
    
    gui.add(settings, 'showTopPositions').onChange(value => {
        topPositions.forEach(point => point.visible = value);
    });
    
    gui.add(settings, 'accumulatorOpacity', 0, 1).onChange(value => {
        accumulatorPoints.forEach(point => point.material.opacity = value);
    });
}

function onWindowResize() {
    camera.aspect = window.innerWidth / window.innerHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(window.innerWidth, window.innerHeight);
}

function animate() {
    requestAnimationFrame(animate);
    controls.update();
    renderer.render(scene, camera);
}

function runAnalysis() {
    // Get input values
    const params = {
        dataSize: parseInt(document.getElementById('dataSize').value),
        accSize: parseInt(document.getElementById('accSize').value),
        raycastSize: parseInt(document.getElementById('raycastSize').value),
        probability: parseFloat(document.getElementById('probability').value)
    };

    // Validate inputs
    if (params.dataSize >= params.accSize) {
        alert('Data size must be smaller than accumulator size');
        return;
    }

    if (params.raycastSize % 2 === 0) {
        alert('Raycast size must be odd');
        return;
    }

    // Clear existing visualization
    clearScene();

    // Run Python script with parameters
    const command = `python run_analysis.py ${params.dataSize} ${params.accSize} ${params.raycastSize} ${params.probability}`;
    
    // Show loading message
    const loadingDiv = document.createElement('div');
    loadingDiv.id = 'loading';
    loadingDiv.style.position = 'absolute';
    loadingDiv.style.top = '50%';
    loadingDiv.style.left = '50%';
    loadingDiv.style.transform = 'translate(-50%, -50%)';
    loadingDiv.style.background = 'rgba(0,0,0,0.7)';
    loadingDiv.style.color = 'white';
    loadingDiv.style.padding = '20px';
    loadingDiv.style.borderRadius = '5px';
    loadingDiv.textContent = 'Running analysis...';
    document.body.appendChild(loadingDiv);

    // Execute Python script
    fetch(`/run?cmd=${encodeURIComponent(command)}`)
        .then(response => {
            if (!response.ok) throw new Error('Analysis failed');
            return fetch('visibility_data.json');
        })
        .then(response => response.json())
        .then(data => {
            createDataPoints(data.dataPoints);
            createAccumulatorPoints(data.accumulatorPoints);
            createTopPositions(data.topPositions);
            document.body.removeChild(loadingDiv);
        })
        .catch(error => {
            console.error('Error:', error);
            alert('Error running analysis. Please check the console for details.');
            document.body.removeChild(loadingDiv);
        });
}

function clearScene() {
    // Remove existing points
    dataPoints.forEach(point => scene.remove(point));
    accumulatorPoints.forEach(point => scene.remove(point));
    topPositions.forEach(point => scene.remove(point));
    
    dataPoints = [];
    accumulatorPoints = [];
    topPositions = [];
} 