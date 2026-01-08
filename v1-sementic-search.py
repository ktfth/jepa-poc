# JEPA + Busca Semântica (Python puro, sem deps)
import random, math

# ---------- util ----------
def tokenize(s): return [w for w in s.lower().replace("?","").replace(".","").split() if w]
def cosine(a,b):
    da=sum(x*x for x in a); db=sum(x*x for x in b)
    if da==0 or db==0: return 0.0
    return sum(x*y for x,y in zip(a,b))/math.sqrt(da*db)

def mse(a,b):
    d=[ai-bi for ai,bi in zip(a,b)]
    return sum(x*x for x in d)/len(d), [2*x/len(d) for x in d]

def tanh(x): return [math.tanh(v) for v in x]
def dtanh(y): return [1.0 - v*v for v in y]  # y=tanh(x)

# ---------- modelo ----------
class Lin:
    def __init__(s, din, dout):
        s.W=[[random.uniform(-0.1,0.1) for _ in range(din)] for _ in range(dout)]
        s.b=[0.0]*dout
    def fwd(s, x):
        s.x=x
        return [sum(wi*xi for wi,xi in zip(w,x))+bi for w,bi in zip(s.W,s.b)]
    def bwd(s, dy, lr):
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

def ema_update(target, online, m=0.99):
    for tl, ol in [(target.l1, online.l1),(target.l2, online.l2)]:
        for o in range(len(tl.W)):
            for i in range(len(tl.W[o])):
                tl.W[o][i]=m*tl.W[o][i]+(1-m)*ol.W[o][i]
            tl.b[o]=m*tl.b[o]+(1-m)*ol.b[o]+(1-m)*0.0

# ---------- embeddings + pooling ----------
def init_emb(vocab_size, d):
    return [[random.uniform(-0.2,0.2) for _ in range(d)] for _ in range(vocab_size)]

def mean_pool(tokens, stoi, E):
    # média dos embeddings das palavras (bag-of-words simples)
    vec=[0.0]*len(E[0]); n=0
    for w in tokens:
        idx=stoi.get(w, 0)
        ew=E[idx]
        for i,v in enumerate(ew): vec[i]+=v
        n+=1
    if n==0: return vec
    return [v/n for v in vec]

# ---------- span masking ----------
def split_context_target(tokens):
    if len(tokens) < 4:
        mid=len(tokens)//2
        return tokens[:mid], tokens[mid:]
    # remove um span interno
    i=random.randint(1, len(tokens)-3)
    j=random.randint(i+1, min(len(tokens)-1, i+3))
    context=tokens[:i] + tokens[j:]
    target=tokens[i:j]
    return context, target

# ---------- corpus (toy) ----------
DOCS = [
 ("como cancelar assinatura", "Para cancelar, vá em Configurações > Assinatura > Cancelar."),
 ("quero reembolso de cobranca indevida", "Solicite estorno no menu Pagamentos > Problemas > Estorno."),
 ("nao consigo fazer login erro senha", "Tente redefinir a senha e verifique seu e-mail."),
 ("meu pedido atrasou prazo ruim", "Verifique o rastreio; se passar do prazo, abrimos ocorrência."),
 ("como alterar endereco de entrega", "Em Pedidos > Detalhes > Alterar endereço, antes do envio."),
 ("preciso segunda via boleto", "A segunda via fica em Pagamentos > Boletos > Gerar novamente."),
 ("aplicativo travando muito", "Atualize o app e limpe o cache; se persistir, envie logs."),
 ("como falar com suporte humano", "Use Ajuda > Falar com atendente (horário comercial)."),
 ("cartao foi recusado pagamento", "Confirme limite, dados e tente outro método de pagamento."),
 ("como mudar plano", "Em Assinatura > Planos, escolha o novo e confirme.")
]

# Gera variações simples para aumentar dados de pré-treino (sem rótulo)
SYN = [
 "como faço para {x}", "preciso de ajuda com {x}", "qual o passo a passo para {x}",
 "nao consigo {x}", "estou com problema em {x}", "erro ao tentar {x}"
]
TOPICS = ["cancelar assinatura","reembolso","fazer login","alterar endereco","segunda via boleto","falar com suporte","pagamento cartao","mudar plano","pedido atrasou","app travando"]

def build_corpus():
    base=[q for q,_ in DOCS] + [a for _,a in DOCS]
    for _ in range(600):
        t=random.choice(TOPICS); p=random.choice(SYN)
        base.append(p.format(x=t))
    return base

def build_vocab(sentences, max_vocab=1000):
    freq={}
    for s in sentences:
        for w in tokenize(s):
            freq[w]=freq.get(w,0)+1
    words=sorted(freq.items(), key=lambda x:-x[1])[:max_vocab-1]
    itos=["<unk>"]+[w for w,_ in words]
    stoi={w:i for i,w in enumerate(itos)}
    return stoi, itos

# ---------- treino JEPA ----------
def pretrain_jepa(sentences, stoi, Demb=24, Z=16, H=32, epochs=8, lr=0.03):
    E=init_emb(len(stoi), Demb)
    enc=MLP(Demb, H, Z)
    tgt=MLP(Demb, H, Z)
    pred=MLP(Z, H, Z)
    for ep in range(epochs):
        random.shuffle(sentences)
        loss_sum=0.0
        for s in sentences:
            toks=tokenize(s)
            c,t=split_context_target(toks)
            vc=mean_pool(c, stoi, E)
            vt=mean_pool(t, stoi, E)

            zc=enc.fwd(vc)
            zt=tgt.fwd(vt)          # stop-grad: não chamamos bwd em tgt
            zhat=pred.fwd(zc)
            loss, dzhat = mse(zhat, zt)
            loss_sum += loss

            dzc = pred.bwd(dzhat, lr)
            enc.bwd(dzc, lr)
            ema_update(tgt, enc, m=0.99)
        print(f"[pretrain] ep={ep} loss={loss_sum/len(sentences):.4f}")
    return E, enc

def encode_text(s, stoi, E, enc):
    v=mean_pool(tokenize(s), stoi, E)
    return enc.fwd(v)

# ---------- busca ----------
def build_index(DOCS, stoi, E, enc):
    idx=[]
    for q,a in DOCS:
        z=encode_text(q, stoi, E, enc)
        idx.append((q,a,z))
    return idx

def search(query, index, stoi, E, enc, k=3):
    zq=encode_text(query, stoi, E, enc)
    scored=[(cosine(zq,z), q,a) for (q,a,z) in index]
    scored.sort(reverse=True, key=lambda x:x[0])
    return scored[:k]

# ---------- demo ----------
def main():
    random.seed(0)
    corpus=build_corpus()
    stoi,_=build_vocab(corpus, max_vocab=800)

    print("Treinando JEPA (texto) ...")
    E, enc = pretrain_jepa(corpus[:], stoi, epochs=8)

    index = build_index(DOCS, stoi, E, enc)

    # baseline aleatório (mesma vocab, embeddings aleatórios, encoder aleatório)
    E0=init_emb(len(stoi), 24)
    enc0=MLP(24, 32, 16)
    index0=build_index(DOCS, stoi, E0, enc0)

    queries=[
        "quero cancelar meu plano agora",
        "cobranca errada preciso estorno",
        "nao entra na conta senha nao funciona",
        "pedido passou do prazo e atrasou",
        "como falar com atendente humano"
    ]

    for q in queries:
        print("\nQUERY:", q)
        print("  JEPA:")
        for sc,qq,aa in search(q, index, stoi, E, enc, k=3):
            print(f"   score={sc:.3f} | {qq} -> {aa}")
        print("  Baseline (aleatório):")
        for sc,qq,aa in search(q, index0, stoi, E0, enc0, k=3):
            print(f"   score={sc:.3f} | {qq} -> {aa}")

if __name__=="__main__":
    main()

