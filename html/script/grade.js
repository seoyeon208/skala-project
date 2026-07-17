function calculateGrades() {
  var subjects = ["HTML", "CSS", "JavaScript"];
  var total = 0;

  for (var i = 0; i < subjects.length; i++) {
    var valid = false;
    while (!valid) {
      var scoreStr = prompt(subjects[i] + " 점수를 입력하세요. (0~100)");

      if (scoreStr === null) {
        return;
      }

      var score = Number(scoreStr);
      if (isNaN(score) || scoreStr.trim() === "") {
        alert("숫자를 입력해주세요.");
      } else if (score < 0 || score > 100) {
        alert("0점부터 100점 사이의 값을 입력해주세요.");
      } else {
        total += score;
        valid = true;
      }
    }
  }

  var average = total / subjects.length;
  var result = average >= 60 ? "합격" : "불합격";

  alert("총점: " + total + "점, 평균: " + average + ", 결과: " + result + "입니다!");
}
