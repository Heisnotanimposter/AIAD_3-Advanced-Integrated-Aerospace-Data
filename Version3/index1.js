// app.js

// Scene, Camera, Renderer
const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(
    45, window.innerWidth / window.innerHeight, 0.1, 1000
);
const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setSize(window.innerWidth, window.innerHeight);
document.body.appendChild(renderer.domElement);

// Controls
const controls = new THREE.OrbitControls(camera, renderer.domElement);
controls.enablePan = false;
controls.enableZoom = false;

// Lighting
const ambientLight = new THREE.AmbientLight(0xffffff, 0.7);
scene.add(ambientLight);

const directionalLight = new THREE.DirectionalLight(0xffffff, 0.6);
directionalLight.position.set(5, 3, 5);
scene.add(directionalLight);

// Earth Geometry and Material
const sphereGeometry = new THREE.SphereGeometry(5, 64, 64);
const earthTexture = new THREE.TextureLoader().load('textures/earth.jpg');
const sphereMaterial = new THREE.MeshStandardMaterial({ map: earthTexture });

const earthMesh = new THREE.Mesh(sphereGeometry, sphereMaterial);
scene.add(earthMesh);

// Camera Position
camera.position.set(0, 0, 15);

// Window Resize Handler
window.addEventListener('resize', () => {
    renderer.setSize(window.innerWidth, window.innerHeight);
    camera.aspect = window.innerWidth / window.innerHeight;
    camera.updateProjectionMatrix();
});

// Zoom Controls
document.getElementById('zoomIn').addEventListener('click', () => {
    camera.position.z -= 1;
});

document.getElementById('zoomOut').addEventListener('click', () => {
    camera.position.z += 1;
});

// Keyboard Controls
document.addEventListener('keydown', (event) => {
    switch (event.key) {
        case 'w':
            earthMesh.rotation.x -= 0.05;
            break;
        case 's':
            earthMesh.rotation.x += 0.05;
            break;
        case 'a':
            earthMesh.rotation.y -= 0.05;
            break;
        case 'd':
            earthMesh.rotation.y += 0.05;
            break;
    }
});

// Raycaster for Interactivity
const raycaster = new THREE.Raycaster();
const mouse = new THREE.Vector2();

function onMouseClick(event) {
    mouse.x = (event.clientX / window.innerWidth) * 2 - 1;
    mouse.y = - (event.clientY / window.innerHeight) * 2 + 1;

    raycaster.setFromCamera(mouse, camera);
    const intersects = raycaster.intersectObjects([earthMesh]);

    if (intersects.length > 0) {
        const point = intersects[0].point;
        const latLon = pointToLatLon(point);
        const country = getCountryAtLatLon(latLon.lat, latLon.lon);

        if (country) {
            highlightCountry(country);
            fetchWikipediaData(country.name);
            fetchEconomicData(country.code);
            generateEconomicSummary(country.name);
        }
    }
}

window.addEventListener('click', onMouseClick, false);

// Convert 3D Point to Latitude and Longitude
function pointToLatLon(point) {
    const radius = 5; // Earth radius in the scene
    const lat = 90 - (Math.acos(point.y / radius)) * 180 / Math.PI;
    const lon = ((270 + Math.atan2(point.x, point.z) * 180 / Math.PI) % 360) - 180;
    return { lat, lon };
}

// Placeholder for Country Detection
function getCountryAtLatLon(lat, lon) {
    // This function should return the country data at the given latitude and longitude
    // For now, return a sample country
    return { name: 'Sample Country', code: 'SC' };
}

// Highlight Country Borders
function highlightCountry(country) {
    // Implement border highlighting using GeoJSON data
    console.log(`Highlighting country: ${country.name}`);
}

// Fetch Wikipedia Data
function fetchWikipediaData(title) {
    fetch(`https://en.wikipedia.org/api/rest_v1/page/summary/${encodeURIComponent(title)}`)
        .then(response => response.json())
        .then(data => {
            document.getElementById('infoTitle').innerText = data.title;
            document.getElementById('infoContent').innerText = data.extract;
            document.getElementById('infoOverlay').style.display = 'block';
        })
        .catch(error => console.error('Error fetching Wikipedia data:', error));
}

// Fetch Economic Data (Placeholder)
function fetchEconomicData(countryCode) {
    const data = {
        gdp: '1.5 Trillion USD',
        population: '50 Million',
        interestRate: '3.5%',
    };
    console.log(`Economic data for ${countryCode}:`, data);
    // Display data in the overlay or use it in charts
}

// Generate Economic Summary (Placeholder)
function generateEconomicSummary(location) {
    const summary = `The economy of ${location} is rapidly growing with a focus on technology and services.`;
    console.log(`Economic summary for ${location}:`, summary);
    // Append summary to the info overlay
    const infoContent = document.getElementById('infoContent');
    infoContent.innerText += `\n\n${summary}`;
}

// Toggle Data Layers
document.getElementById('toggleEconomicData').addEventListener('click', () => {
    // Toggle economic data visualization
    console.log('Toggled Economic Data Layer');
});

document.getElementById('toggleTourismData').addEventListener('click', () => {
    // Toggle tourism data visualization
    console.log('Toggled Tourism Data Layer');
});

// Animation Loop
function animate() {
    requestAnimationFrame(animate);
    renderer.render(scene, camera);
}

animate();
// Load and Parse GeoJSON Data
const countryBorders = new THREE.Group();
scene.add(countryBorders);

const loader = new THREE.FileLoader();
loader.load('data/countries.geojson', (data) => {
    const geoData = JSON.parse(data);
    geoData.features.forEach((feature) => {
        const coordinates = feature.geometry.coordinates;
        // Convert coordinates to 3D points and create lines
        // Add lines to countryBorders group
    });
});
function latLonToVector3(lat, lon, radius) {
    const phi = (90 - lat) * (Math.PI / 180);
    const theta = (lon + 180) * (Math.PI / 180);

    const x = - (radius * Math.sin(phi) * Math.cos(theta));
    const y = radius * Math.cos(phi);
    const z = radius * Math.sin(phi) * Math.sin(theta);

    return new THREE.Vector3(x, y, z);
}
// Close Overlay on Click Outside
window.addEventListener('click', (event) => {
    if (!event.target.closest('#infoOverlay')) {
        document.getElementById('infoOverlay').style.display = 'none';
    }
});
function fetchEconomicData(countryCode) {
    // Use fetch to call real APIs
    // Parse and display data
}
