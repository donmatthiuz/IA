
def f1_score(ytest, ypredichas):
  TP = sum((yt == 1 and yp == 1) for yt, yp in zip(ytest, ypredichas))
  FP = sum((yt == 0 and yp == 1) for yt, yp in zip(ytest, ypredichas))
  FN = sum((yt == 1 and yp == 0) for yt, yp in zip(ytest, ypredichas))

  precision = TP / (TP + FP) if (TP + FP) > 0 else 0
  recall = TP / (TP + FN) if (TP + FN) > 0 else 0

  f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

  return f1

