# JEPA mínimo em Python puro (sem deps). Rode: python jepa_min.py
import random, math

# --------- ops vetoriais ----------
def tanh(x): return [math.tanh(v) for v in x]
def dtanh(y): return [1.0 - v*v for v in y]  # y = tanh(x)
def mse(a,b):
    d=[ai-bi for ai,bi in zip(a,b)]
    return sum(x*x for x in d)/len(d), [2*x/len(d) for x in d]  # (loss, grad wrt a)

# --------- camadas ----------
class Lin:
    def __init__(s, din, dout):
        s.W=[[random.uniform(-0.2,0.2) for _ in range(din)] for _ in range(dout)]
        s.b=[0.0]*dout
    def fwd(s, x):
        s.x=x
        return [sum(wi*xi for wi,xi in zip(w,x))+bi for w,bi in zip(s.W,s.b)]
    def bwd(s, dy, lr):
        # grad W,b e dx; atualiza SGD in-place
        dx=[0.0]*len(s.x)
        for o,(w,go) in enumerate(zip(s.W,dy)):
            s.b[o]-=lr*go
            for i,xi in enumerate(s.x):
                dx[i]+=w[i]*go
                w[i]-=lr*go*xi
        return dx

class MLP:
    def __init__(s, din, dh, dout):
        s.l1, s.l2 = Lin(din,dh), Lin(dh,dout)
    def fwd(s, x):
        s.h=tanh(s.l1.fwd(x))
        return s.l2.fwd(s.h)
    def bwd(s, dy, lr):
        dh=s.l2.bwd(dy, lr)
        dh=[g*t for g,t in zip(dh, dtanh(s.h))]
        return s.l1.bwd(dh, lr)

# --------- util: EMA do target encoder ----------
def ema_update(target, online, m=0.995):
    for tl, ol in [(target.l1, online.l1),(target.l2, online.l2)]:
        for o in range(len(tl.W)):
            for i in range(len(tl.W[o])):
                tl.W[o][i]=m*tl.W[o][i]+(1-m)*ol.W[o][i]
            tl.b[o]=m*tl.b[o]+(1-m)*ol.b[o]

# --------- dataset sintético ----------
# x tem D dims; context = primeiras Dc; target = últimas Dt
def make_batch(n, D):
    # "mundo" gerador: alguns fatores latentes e ruído
    out=[]
    for _ in range(n):
        a=random.uniform(-1,1); b=random.uniform(-1,1)
        x=[a, b, a*b, a*a-b*b, math.sin(a), math.cos(b), 0.3*a+0.7*b]
        x += [random.uniform(-0.05,0.05) for _ in range(max(0,D-len(x)))]
        out.append(x[:D])
    return out

# --------- treino ----------
def run():
    random.seed(0)
    D, Dc, Dt = 8, 4, 4
    Z, H = 8, 16
    enc = MLP(Dc, H, Z)         # online/context encoder
    tgt = MLP(Dt, H, Z)         # target encoder (EMA)
    pred = MLP(Z,  H, Z)        # predictor: zc -> zt_hat
    head = Lin(Z, 1)            # downstream head simples

    # ----- Pré-treino JEPA -----
    lr=0.03
    for ep in range(6):
        xs=make_batch(64, D)
        loss_sum=0.0
        for x in xs:
            c, t = x[:Dc], x[Dc:]
            zc = enc.fwd(c)
            zt = tgt.fwd(t)          # sem grad (não chamamos bwd)
            zt_hat = pred.fwd(zc)
            loss, dzt_hat = mse(zt_hat, zt)
            loss_sum += loss
            dzc = pred.bwd(dzt_hat, lr)
            enc.bwd(dzc, lr)
            ema_update(tgt, enc, m=0.99)  # alvo acompanha online lentamente
        print(f"[pretrain] ep={ep} loss={loss_sum/len(xs):.4f}")

    # ----- Downstream supervisionado (toy) -----
    # rótulo: 1 se soma de x > 0, senão 0
    lr=0.05
    for ep in range(6):
        xs=make_batch(64, D)
        loss_sum=0.0
        for x in xs:
            y = 1.0 if sum(x) > 0 else 0.0
            z = enc.fwd(x[:Dc])
            yhat = head.fwd(z)[0]
            loss = (yhat - y)**2
            loss_sum += loss
            dy = [2*(yhat - y)]
            dz = head.bwd(dy, lr)
            enc.bwd(dz, lr*0.2)   # opcional: fine-tune leve
        print(f"[downstream] ep={ep} mse={loss_sum/len(xs):.4f}")

if __name__ == "__main__":
    run()

