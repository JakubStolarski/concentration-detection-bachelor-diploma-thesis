// JavaScript source code
function yesnoCheck() {
    var x = document.getElementById("ModeSelect").value;

    if (x == 'Video') {
        document.getElementById('IfYes').style.display = 'block';
    }
    else {
        document.getElementById('IfYes').style.display = 'none';
    }
}