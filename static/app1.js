document.getElementById("input-text").addEventListener("input", function () {
    const text = this.value;

    if (text) {
        fetch("/translate", {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
            },
            body: JSON.stringify({
                sentence: text
            })
        })
        .then(response => response.json())
        .then(data => {
            const outputText = document.getElementById('output-text');
            outputText.innerText = data.translation;
            if (data.translation.trim()) {
                outputText.classList.remove('empty');
            } else {
                outputText.classList.add('empty');
            }
        });
    } else {
        document.getElementById("output-text").innerText = "";
        document.getElementById("output-text").classList.add('empty');
    }
});
