$(function () {
  async function classifyImage(imageElement) {
    try {
      const response = await fetch("/classify", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ imageName: imageElement.id + ".jpg" }),
      });
      const results = await response.json();
      displayResults(results, imageElement.id);
    } catch (error) {
      console.log("Error:", error);
    }
  }

  function displayResults(results, imageId) {
    const $resultDiv = $("#" + imageId + "-result");
    $resultDiv.empty();

    const $list = $("<ol>");
    results.forEach(function (result) {
      const $item = $("<li>").text(result.class + ": " + result.score + "%");
      $list.append($item);
    });

    $resultDiv.append($list);
  }

  $("#classify-btn").click(async function () {
    const imageElements = [
      document.getElementById("diagnose1"),
      document.getElementById("diagnose2"),
      document.getElementById("diagnose3"),
    ];

    for (const imageElement of imageElements) {
      await classifyImage(imageElement);
    }
  });
});
