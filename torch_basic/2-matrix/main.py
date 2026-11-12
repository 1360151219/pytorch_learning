import torch


def main():
  x = torch.tensor([1.0,2.0], requires_grad=True)
  w = torch.tensor([3.0,4.0], requires_grad=True)
  y = x[0]*w[0] + x[1]*w[1]
  loss = y**2
  loss.backward()
  print(y) # y = xw^T = 1*3+2*4 = 11
  print(w.grad) # 

def main2():
  w = torch.tensor(2.0, requires_grad=True)
  a = w ** 2
  b = a.detach()
  z = a + 3 * b
  z.backward()
  # 1*2*2+ (3*0*2*2 切断)
  print(w.grad)
  print(a.requires_grad)
  print(b.requires_grad)


def main3():
  x = torch.tensor([[1.0, 0.0],
                  [0.0, 1.0],
                  [1.0, 1.0]])
  # target
  y = torch.tensor([[2.0],
                    [1.0],
                    [3.0]])
  w = torch.tensor([[1.0],
                    [2.0]], requires_grad=True)

  optimizer = torch.optim.SGD([w], lr=0.25)
  optimizer.zero_grad()
  # predict
  y_hat = x @ w
  r = y_hat - y
  loss = r.T @ r

  loss.backward()
  optimizer.step()

  print(w.grad)
  print(w)

if __name__ == "__main__":
  # main()
  main3()