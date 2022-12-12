// JavaScript source code
function open_video() {
    var x = document.forms["InputSelect"]["Camera ID"].value
    if (x) {
        document.getElementById("streaming").style.display = 'block';
    }
}