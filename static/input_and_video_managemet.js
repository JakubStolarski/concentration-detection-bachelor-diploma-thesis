// JavaScript source code
function yesnoCheck() {
    var x = document.getElementById("ModeSelect").value;

    if (x == 'Showcase') {
        document.getElementById('IfYes').style.display = 'block';
    }
    else {
        document.getElementById('IfYes').style.display = 'none';
    }
}

function select_to_input() {
    var x = document.getElementById("ModeSelect").value;
    switch (x) {
        case "Configuration":
            document.getElementById('Mode').value = '0'
        case "Run":
            document.getElementById('Mode').value = '1'
        case "Video":
            document.getElementById('Mode').value = '2'
        case "Showcase":
            document.getElementById('Mode').value = '3'
    }
}

function open_video() {
    document.getElementById("streaming").style.display = 'block';
}