// JavaScript source code
function select_to_input() {
    var x = document.getElementById("ModeSelect").value;
    switch (x) {
        case "Configuration":
            document.getElementById('Mode').value = '0'
            break;
        case "Run":
            document.getElementById('Mode').value = '1'
            break;
        case "Video":
            document.getElementById('Mode').value = '2'
            break;
    }
}