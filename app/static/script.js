
document.querySelectorAll(".tab").forEach(tab => {
  tab.addEventListener("click", () => {
    // deactivate all
    document.querySelectorAll(".tab").forEach(t => t.classList.remove("active"));
    document.querySelectorAll(".tab-content").forEach(c => c.classList.remove("active"));
    
    // activate selected
    tab.classList.add("active");
    const target = tab.dataset.target;
    document.getElementById(target).classList.add("active");
  });
});


const tooltips = document.querySelectorAll('.tooltip');

tooltips.forEach((tooltip) => {
  const closeBtn = tooltip.querySelector('.close-btn');

  // Show tooltip on icon click
  tooltip.addEventListener('click', (e) => {
    // Close other tooltips first
    tooltips.forEach(t => t.classList.remove('active'));
    tooltip.classList.add('active');
    e.stopPropagation();
  });

  // Close when clicking the '×'
  closeBtn.addEventListener('click', (e) => {
    tooltip.classList.remove('active');
    e.stopPropagation();
  });
});

// Close tooltip if clicking anywhere else
document.addEventListener('click', () => {
  tooltips.forEach(t => t.classList.remove('active'));
});

const visualForm = document.getElementById("visual-form");
const visualOutput = document.getElementById("visual-output");
const downloadBtn = document.getElementById("download-btn"); // make sure this element exists in HTML

visualForm.addEventListener("submit", async (e) => {
  e.preventDefault();
  
  visualOutput.innerHTML = "<p>Generating GIF...</p>"; // optional loading message

  const formData = new FormData(visualForm);

  const response = await fetch("/visualize", {
    method: "POST",
    body: formData
  });

  if (!response.ok) {
    visualOutput.innerHTML = "<p style='color:red;'>Error generating visualization</p>";
    return;
  }

  const data = await response.json();
  const gifUrl = data.gif_url;

  // Embed GIF in the page
  visualOutput.innerHTML = `
    <p>Generated GIF:</p>
    <img src="${gifUrl}" alt="Visualization GIF" style="max-width:100%; border-radius:10px;">
    <br>
    <a id="download-btn" href="${gifUrl}" download>Download GIF</a>
  `;
});
