document.addEventListener("DOMContentLoaded", function () {
  const ctx = document.getElementById("umap-plot").getContext("2d");
  const tooltipContainer = document.getElementById("tooltip-container");
  let chart;

  const categoryBtn = document.getElementById("categoryBtn");
  const selectedCategory = document.getElementById("selectedCategory");
  const dropdownContainer = document.querySelector(".dropdown-content");

  // Toggle the 'show' class when the category button is clicked
  categoryBtn.addEventListener("click", function (event) {
    event.preventDefault(); // Prevent the default behavior (page jump)
    dropdownContainer.classList.toggle("show");
  });

  // Handle clicks on dropdown items
  dropdownContainer.addEventListener("click", handleDropdownClick);

  // Handle clicks outside the dropdown to close it
  document.addEventListener("click", function (event) {
    const isClickInsideDropdown = dropdownContainer.contains(event.target);
    const isClickInsideCategoryBtn = categoryBtn.contains(event.target);
    if (!isClickInsideDropdown && !isClickInsideCategoryBtn) {
      dropdownContainer.classList.remove("show");
    }
  });

  function handleDropdownClick(event) {
    event.preventDefault(); // Prevent the default behavior (page jump)

    if (event.target.tagName === "A") {
      const category = event.target.dataset.category;
      selectedCategory.textContent = event.target.textContent;
      hideTooltip();
      loadUMAPData(category);
      dropdownContainer.classList.remove("show"); // Hide dropdown after selecting an option
    }
  }

  function loadUMAPData(category) {
    if (chart) {
      console.log("Destroying existing chart");
      chart.destroy();
    }

    fetch(`umap_data_${category}.json`)
      .then((response) => response.json())
      .then((umapData) => {
        return fetch(`cluster_data_${category}.json`)
          .then((response) => response.json())
          .then((clusterData) => ({ umapData, clusterData }));
      })
      .then((data) => {
        const embeddings = data.umapData.embeddings;
        const additionalInfo = data.umapData.additional_info;
        const clusters = data.clusterData.clusters;

        const bubbleData = embeddings.map((point, index) => ({
          x: point[0],
          y: point[1],
          r: 10,
          title: additionalInfo[index].title,
          link: additionalInfo[index].arxiv_id
            ? `https://arxiv.org/abs/${additionalInfo[index].arxiv_id}`
            : null,
          cluster: clusters[index],
        }));

        console.log("Loaded Bubble Data:", bubbleData);

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
                  const clusterColorMap = {
                    0: "#1f77b4",
                    1: "#ff7f0e",
                    2: "#2ca02c",
                    3: "#d62728",
                    4: "#9467bd",
                  };
                  const cluster =
                    context.dataset.data[context.dataIndex].cluster;
                  return clusterColorMap[cluster] || "#000000";
                },
                borderColor: "#FFFFFF",
                borderWidth: 1,
              },
            },
          },
        });

        console.log("New chart created");

        attachChartListeners();
      })
      .catch((error) =>
        console.error("Error loading UMAP and cluster data:", error),
      );
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

  loadUMAPData("Computer_Science");
});
