def correct_rating(x):
  if x < 1:
    return 0
  elif (x >= 1) & (x < 1.5):
    return 1
  elif (x >= 1.5) & (x < 2.5):
    return 2
  elif (x >= 2.5) & (x < 3.5):
    return 3
  elif (x >= 3.5) & (x < 4.5):
    return 4
  else:       #(x >= 4.5):
    return 5