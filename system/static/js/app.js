// 页面级交互：填充示例检索词、选参考图、带上修改文本跳转结果页。
document.querySelectorAll('.chip[data-fill]').forEach(function (chip) {
  chip.addEventListener('click', function () {
    var box = document.getElementById('q');
    if (box) { box.value = chip.dataset.fill; box.focus(); }
  });
});

var picker = document.getElementById('file');
if (picker) {
  picker.addEventListener('change', function () {
    if (picker.files.length) { document.getElementById('up').submit(); }
  });
}

document.querySelectorAll('#refgrid .tile[data-item]').forEach(function (tile) {
  tile.addEventListener('click', function (event) {
    event.preventDefault();
    var mod = document.getElementById('mod');
    var text = mod && mod.value.trim();
    var url = '/results?mode=' + (text ? 'fusion' : 'image') +
              '&item=' + encodeURIComponent(tile.dataset.item);
    if (text) { url += '&q=' + encodeURIComponent(text); }
    window.location.href = url;
  });
});

var paste = document.getElementById('paste');
if (paste) {
  paste.addEventListener('click', function () {
    var url = window.prompt('粘贴图片 URL');
    if (url) { window.location.href = '/results?mode=image&url=' + encodeURIComponent(url); }
  });
}
