function playUpDownGame() {
  var computerNum = Math.floor(Math.random() * 50) + 1;
  var count = 0;

  while (true) {
    var inputStr = prompt("1부터 50 사이의 숫자를 맞춰보세요!");

    if (inputStr === null) {
      break;
    }

    var userNum = Number(inputStr);
    count++;

    if (userNum > computerNum) {
      alert("Down!");
    } else if (userNum < computerNum) {
      alert("Up!");
    } else if (userNum === computerNum) {
      alert("축하합니다! " + count + "번 만에 맞추셨습니다.");
      break;
    } else {
      alert("올바른 숫자를 입력해주세요.");
      count--;
    }
  }
}
