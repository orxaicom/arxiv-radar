document.addEventListener("DOMContentLoaded", function () {
  const ctx = document.getElementById("umap-plot").getContext("2d");
  const tooltipContainer = document.getElementById("tooltip-container");
  const searchInput = document.getElementById("searchInput");
  let chart;
  let bubbleData;

  const categoryBtn = document.getElementById("categoryBtn");
  const selectedCategory = document.getElementById("selectedCategory");
  const dropdownContainer = document.querySelector(".dropdown-content");

  categoryBtn.addEventListener("click", function (event) {
    event.preventDefault();
    dropdownContainer.classList.toggle("show");
  });

  dropdownContainer.addEventListener("click", handleDropdownClick);

  document.addEventListener("click", function (event) {
    const isClickInsideDropdown = dropdownContainer.contains(event.target);
    const isClickInsideCategoryBtn = categoryBtn.contains(event.target);
    if (!isClickInsideDropdown && !isClickInsideCategoryBtn) {
      dropdownContainer.classList.remove("show");
    }
  });

  searchInput.addEventListener("input", handleSearch);

  function handleDropdownClick(event) {
    event.preventDefault();

    if (event.target.tagName === "A") {
      const category = event.target.dataset.category;
      selectedCategory.textContent = event.target.textContent;
      hideTooltip();
      loadUMAPData(category);
      dropdownContainer.classList.remove("show");
    }
  }

  function loadUMAPData(category) {
    if (chart) {
      chart.destroy();
    }

    Promise.all([
      fetch(`umap_data_${category}.json`).then(response => response.json()),
      fetch(`cluster_data_${category}.json`).then(response => response.json())
    ])
    .then(([umapData, clusterData]) => {
      const embeddings = umapData.embeddings;
      const additionalInfo = umapData.additional_info;
      const clusters = clusterData.clusters;

      bubbleData = embeddings.map((point, index) => ({
        x: point[0],
        y: point[1],
        r: 10,
        title: additionalInfo[index].title,
        link: additionalInfo[index].arxiv_id
          ? `https://arxiv.org/abs/${additionalInfo[index].arxiv_id}`
          : null,
        cluster: clusters[index],
        opacity: 1, // Add opacity property
      }));

      createChart();
      attachChartListeners();
    })
    .catch((error) => console.error("Error loading UMAP and cluster data:", error));
  }

  function createChart() {
    chart = new Chart(ctx, {
      type: "bubble",
      data: {
        datasets: [
          {
            label: "UMAP Plot",
            data: bubbleData,
          },
        ],
      },
      options: {
        scales: {
          x: {
            title: { display: true, text: "UMAP Component 1" },
            grid: { color: "rgba(255,255,255,0.2)" },
          },
          y: {
            title: { display: true, text: "UMAP Component 2" },
            grid: { color: "rgba(255,255,255,0.2)" },
          },
        },
        plugins: {
          legend: { display: false },
          tooltip: {
            enabled: false,
          },
        },
        elements: {
          point: {
            backgroundColor: (context) => {
              const dataPoint = context.dataset.data[context.dataIndex];
              const color = getColorForCluster(dataPoint.cluster);
              return `rgba(${getRGBFromHex(color)}, ${dataPoint.opacity})`;
            },
            borderColor: (context) => {
              const dataPoint = context.dataset.data[context.dataIndex];
              return `rgba(255, 255, 255, ${dataPoint.opacity})`;
            },
            borderWidth: 1,
          },
        },
        animation: {
          duration: 0 // Disable animations for smoother updates
        }
      },
    });
  }

  function attachChartListeners() {
    ctx.canvas.removeEventListener("mousemove", handleMouseMove);
    ctx.canvas.removeEventListener("click", handleMouseClick);

    ctx.canvas.addEventListener("mousemove", handleMouseMove);
    ctx.canvas.addEventListener("click", handleMouseClick);
  }

  function handleMouseMove(event) {
    const xOffset = 20;
    const yOffset = 20;

    if (chart) {
      const activePoint = chart.getElementsAtEventForMode(
        event,
        "nearest",
        { intersect: true },
        false,
      )[0];

      if (activePoint) {
        const datasetIndex = activePoint.datasetIndex;
        const dataIndex = activePoint.index;
        const info = chart.data.datasets[datasetIndex].data[dataIndex];
        showTooltip(
          event.pageX + xOffset,
          event.pageY - yOffset,
          info.title,
        );
      } else {
        hideTooltip();
      }
    }
  }

  function handleMouseClick(event) {
    if (chart) {
      const activePoint = chart.getElementsAtEventForMode(
        event,
        "nearest",
        { intersect: true },
        false,
      )[0];

      if (activePoint) {
        const datasetIndex = activePoint.datasetIndex;
        const dataIndex = activePoint.index;
        const info = chart.data.datasets[datasetIndex].data[dataIndex];

        if (info.link) {
          window.open(info.link, "_blank");
        }
      }
    }
  }

  function showTooltip(x, y, title) {
    tooltipContainer.innerHTML = `<span>${title}</span>`;
    tooltipContainer.style.left = x + "px";
    tooltipContainer.style.top = y - tooltipContainer.clientHeight + "px";
    tooltipContainer.style.display = "block";
  }

  function hideTooltip() {
    tooltipContainer.style.display = "none";
  }

  function handleSearch() {
    const searchTerm = searchInput.value.toLowerCase();
    
    if (chart) {
      bubbleData.forEach((dataPoint) => {
        const title = dataPoint.title.toLowerCase();
        dataPoint.opacity = title.includes(searchTerm) || searchTerm === "" ? 1 : 0.2;
      });
      
      chart.update();
    }
  }

  function getColorForCluster(cluster) {
    const clusterColorMap = {
      0: "#1f77b4",
      1: "#ff7f0e",
      2: "#2ca02c",
      3: "#d62728",
      4: "#9467bd",
    };
    return clusterColorMap[cluster] || "#000000";
  }

  function getRGBFromHex(hex) {
    const r = parseInt(hex.slice(1, 3), 16);
    const g = parseInt(hex.slice(3, 5), 16);
    const b = parseInt(hex.slice(5, 7), 16);
    return `${r}, ${g}, ${b}`;
  }

  loadUMAPData("Computer_Science");
});
