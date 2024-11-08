changequote(`{{', `}}')
define({{CURRENT_DATE}}, {{esyscmd(date -u +"%a %b %d %R %Z" | tr -d '\n')}})
document.addEventListener("DOMContentLoaded", function () {
  const ctx = document.getElementById("umap-plot").getContext("2d");
  const tooltipContainer = document.getElementById("tooltip-container");
  const searchInput = document.getElementById("searchInput");
  const categoryCheckboxesContainer = document.getElementById("categoryCheckboxes");
  const checkboxContainer = document.getElementById("checkboxContainer");
  const selectAllBtn = document.getElementById("selectAllBtn");
  const selectNoneBtn = document.getElementById("selectNoneBtn");
  let chart;
  let bubbleData;
  let categories = new Set();
  let isFirstLoad = true;

  const categoryBtn = document.getElementById("categoryBtn");
  const selectedCategory = document.getElementById("selectedCategory");
  const categoryDropdownContainer = document.querySelector(".dropdown-content");
  const filterBtn = document.getElementById("filterBtn");

  categoryBtn.addEventListener("click", function (event) {
    event.preventDefault();
    categoryDropdownContainer.classList.toggle("show");
  });

  filterBtn.addEventListener("click", function (event) {
    event.preventDefault();
    categoryCheckboxesContainer.classList.toggle("show");
  });

  categoryDropdownContainer.addEventListener("click", handleDropdownClick);

  document.addEventListener("click", function (event) {
    const isClickInsideCategoryDropdown = categoryDropdownContainer.contains(event.target);
    const isClickInsideCategoryBtn = categoryBtn.contains(event.target);
    const isClickInsideFilterDropdown = categoryCheckboxesContainer.contains(event.target);
    const isClickInsideFilterBtn = filterBtn.contains(event.target);

    if (!isClickInsideCategoryDropdown && !isClickInsideCategoryBtn) {
      categoryDropdownContainer.classList.remove("show");
    }

    if (!isClickInsideFilterDropdown && !isClickInsideFilterBtn) {
      categoryCheckboxesContainer.classList.remove("show");
    }
  });

  searchInput.addEventListener("input", handleSearch);

  selectAllBtn.addEventListener("click", function() {
    const checkboxes = checkboxContainer.querySelectorAll('input[type="checkbox"]');
    checkboxes.forEach(checkbox => checkbox.checked = true);
    handleCategoryFilter();
  });

  selectNoneBtn.addEventListener("click", function() {
    const checkboxes = checkboxContainer.querySelectorAll('input[type="checkbox"]');
    checkboxes.forEach(checkbox => checkbox.checked = false);
    handleCategoryFilter();
  });

  function handleDropdownClick(event) {
    event.preventDefault();

    if (event.target.tagName === "A") {
      const category = event.target.dataset.category;
      selectedCategory.textContent = event.target.textContent;
      hideTooltip();
      loadUMAPData(category);
      categoryDropdownContainer.classList.remove("show");
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

      categories.clear();
      bubbleData = embeddings.map((point, index) => {
        const categoryList = additionalInfo[index].categories.split(',');
        categoryList.forEach(cat => categories.add(cat.trim()));
        return {
          x: point[0],
          y: point[1],
          r: 10,
          title: additionalInfo[index].title,
          link: additionalInfo[index].arxiv_id
            ? `https://arxiv.org/abs/${additionalInfo[index].arxiv_id}`
            : null,
          cluster: clusters[index],
          opacity: 1,
          relevant: true,
          categories: categoryList,
        };
      });

      createCategoryCheckboxes();
      createChart();
      isFirstLoad = false;
    })
    .catch((error) => console.error("Error loading UMAP and cluster data:", error));
  }

  function createCategoryCheckboxes() {
    checkboxContainer.innerHTML = '';
    categories.forEach(category => {
      const checkboxDiv = document.createElement('div');
      checkboxDiv.classList.add('category-checkbox');
      
      const checkbox = document.createElement('input');
      checkbox.type = 'checkbox';
      checkbox.id = `category-${category}`;
      checkbox.checked = true;
      checkbox.addEventListener('change', handleCategoryFilter);

      const label = document.createElement('label');
      label.htmlFor = `category-${category}`;
      label.textContent = category;

      checkboxDiv.appendChild(checkbox);
      checkboxDiv.appendChild(label);
      checkboxContainer.appendChild(checkboxDiv);
    });
  }

  function createChart() {
    if (ctx.canvas) {
      ctx.canvas.removeEventListener("mousemove", handleMouseMove);
      ctx.canvas.removeEventListener("click", handleMouseClick);
    }

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
        responsive: true,
        maintainAspectRatio: false,
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
          title: {
            display: true,
            text: "Updated on CURRENT_DATE. No new papers on arXiv on Sat & Sun",
          },
        },
        elements: {
          point: {
            backgroundColor: (context) => {
              const dataPoint = context.dataset.data[context.dataIndex];
              const color = dataPoint.relevant ? getColorForCluster(dataPoint.cluster) : "#808080";
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
          duration: 800,
          easing: 'easeInOutQuad',
          onComplete: function() {
            setTimeout(() => {
              attachChartListeners();
            }, 100);
          }
        }
      },
    });
  }

  function attachChartListeners() {
    if (!chart || !ctx.canvas) return;
    
    ctx.canvas.removeEventListener("mousemove", handleMouseMove);
    ctx.canvas.removeEventListener("click", handleMouseClick);

    ctx.canvas.addEventListener("mousemove", handleMouseMove);
    ctx.canvas.addEventListener("click", handleMouseClick);
  }

  function handleMouseMove(event) {
    if (!chart || !chart.canvas) return;
    
    const elements = chart.getElementsAtEventForMode(
      event,
      "nearest",
      { intersect: true },
      false
    );

    if (elements?.length) {
      const activePoint = elements[0];
      const info = chart.data.datasets[activePoint.datasetIndex].data[activePoint.index];
      showTooltip(event.pageX + 20, event.pageY - 20, info.title);
    } else {
      hideTooltip();
    }
  }

function handleMouseClick(event) {
    if (!chart || !chart.canvas) return;

    const elements = chart.getElementsAtEventForMode(
      event,
      "nearest",
      { intersect: true },
      false
    );

    if (elements?.length) {
      const activePoint = elements[0];
      const info = chart.data.datasets[activePoint.datasetIndex].data[activePoint.index];
      if (info.link) {
        window.open(info.link, "_blank");
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
    updateRelevance();
  }

  function handleCategoryFilter() {
    updateRelevance();
  }

  function updateRelevance() {
    const searchTerm = searchInput.value.toLowerCase();
    const selectedCategories = Array.from(checkboxContainer.querySelectorAll('input:checked'))
      .map(checkbox => checkbox.id.replace('category-', ''));
    
    if (chart) {
      bubbleData.forEach((dataPoint) => {
        const title = dataPoint.title.toLowerCase();
        const categoryMatch = dataPoint.categories.some(cat => selectedCategories.includes(cat));
        dataPoint.relevant = (title.includes(searchTerm) || searchTerm === "") && categoryMatch;
        dataPoint.opacity = dataPoint.relevant ? 1 : 0.2;
      });
      
      chart.options.animation.duration = 0;
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
