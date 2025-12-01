/* ------------------------------
   TAB SWITCHING
--------------------------------*/

// Tab switching functionality
    const tabButtons = document.querySelectorAll('.tab-button');
    const tabContents = document.querySelectorAll('.tab-content');

    tabButtons.forEach(button => {
        button.addEventListener('click', () => {
            const tabId = button.getAttribute('data-tab');

            // Remove active class from all buttons and contents
            tabButtons.forEach(btn => btn.classList.remove('active'));
            tabContents.forEach(content => content.classList.remove('active'));

            // Add active class to clicked button and corresponding content
            button.classList.add('active');
            document.getElementById(tabId).classList.add('active');
        });
    });

/* ============================================================
   UNIVERSAL API HANDLER FOR ALL FORMS
============================================================ */
async function handleFormUpload(form) {
    const formData = new FormData(form);
    const endpoint = form.getAttribute("action");

    const statusBox = form.querySelector(".status-box");
    if (statusBox) statusBox.innerHTML = "<p>Processing...</p>";

    try {
        const response = await fetch(endpoint, {
            method: "POST",
            body: formData
        });

        if (!response.ok) {
            if (statusBox) statusBox.innerHTML = `<p style="color:red;">Server error</p>`;
            return;
        }

        const blob = await response.blob();
        const url = URL.createObjectURL(blob);

        // Auto-download
        const a = document.createElement("a");
        a.href = url;

        // Infer filename
        const cd = response.headers.get("Content-Disposition");
        let filename = "events.txt";
        if (cd) {
            const match = cd.match(/filename="(.+)"/);
            if (match) filename = match[1];
        }
        a.download = filename;
        a.click();

        if (statusBox) statusBox.innerHTML = `<p>✓ Completed. File downloaded.</p>`;

    } catch (err) {
        console.error(err);
        if (statusBox) statusBox.innerHTML = `<p style="color:red;">Unexpected error</p>`;
    }
}

/* ============================================================
   VISUALIZATION (Generates GIF internally)
============================================================ */
const visualForm = document.getElementById("visual-form");
const visualOutput = document.getElementById("visual-output");

visualForm.addEventListener("submit", async (e) => {
    e.preventDefault();

    visualOutput.innerHTML = "<p>Generating GIF…</p>";

    const formData = new FormData(visualForm);

    const response = await fetch("/api/visualize", {
        method: "POST",
        body: formData
    });

    if (!response.ok) {
        visualOutput.innerHTML = "<p style='color:red;'>Error during visualization</p>";
        return;
    }

    const data = await response.json();

    visualOutput.innerHTML = `
    <p>Generated GIF:</p>
    <img src="${data.gif_url}" style="max-width:100%; border-radius:10px;">
    <br>
    <a href="${data.gif_url}" download="visualization.gif">Download GIF</a>
  `;
});


/* ============================================================
   JSON-BASED API CALL FOR LIVE PROCESSING (Already used)
============================================================ */
async function sendEventsToServer(events, width = 320, height = 240) {
    const payload = { events, width, height, blur_faces: true };

    const res = await fetch('/api/process_events', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
    });

    return await res.json();
}
