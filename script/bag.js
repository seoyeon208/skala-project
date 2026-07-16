function showMyBag() {
  var myBag = [
    { name: "노트북", count: 1 },
    { name: "펜", count: 3 },
    { name: "노트", count: 2 },
    { name: "이어폰", count: 1 },
    { name: "지갑", count: 1 }
  ];

  var result = "What's in my bag?\n\n";

  for (var i = 0; i < myBag.length; i++) {
    result += "🌀" + myBag[i].name + myBag[i].count + "개\n";
  }

  alert(result);
}
