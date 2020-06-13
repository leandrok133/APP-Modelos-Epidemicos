import streamlit as st
import plotly.graph_objects as go
import numpy as np

# Definir as funções

#======Para SIR e SIRS
def dSdt (S,I,R,b,beta,d,sigma):
    N = 1.0*S+I+R
    return b*N - beta*S*I/N - d*S + sigma*R

def dIdt (S,I,R,beta,d,delta,gama):
    N = 1.0*S+I+R
    return beta*S*I/N - d*I - delta*I - gama*I

def dRdt (I,R,d,gama,sigma):
    return 1.0*(gama*I - d*R - sigma*R)

def RK4 (dt, Si, Ii, Ri, b, beta, d, delta, sigma, gama):
    
    KS1 = dSdt(Si, Ii, Ri, b, beta, d, sigma)
    KI1 = dIdt(Si, Ii, Ri, beta, d, delta, gama)
    KR1 = dRdt(Ii, Ri, d, gama, sigma)
    
    KS2 = dSdt(Si+dt*KS1/2, Ii+dt*KI1/2, Ri+dt*KR1/2, b, beta, d, sigma)
    KI2 = dIdt(Si+dt*KS1/2, Ii+dt*KI1/2, Ri+dt*KR1/2, beta, d, delta, gama)
    KR2 = dRdt(Ii+dt*KI1/2, Ri+dt*KR1/2, d, gama, sigma)
    
    KS3 = dSdt(Si+dt*KS2/2, Ii+dt*KI2/2, Ri+dt*KR2/2, b, beta, d, sigma)
    KI3 = dIdt(Si+dt*KS2/2, Ii+dt*KI2/2, Ri+dt*KR2/2, beta, d, delta, gama)
    KR3 = dRdt(Ii+dt*KI2/2, Ri+dt*KR2/2, d, gama, sigma)
    
    KS4 = dSdt(Si+dt*KS3, Ii+dt*KI3, Ri+dt*KR3, b, beta, d, sigma)
    KI4 = dIdt(Si+dt*KS3, Ii+dt*KI3, Ri+dt*KR3, beta, d, delta, gama)
    KR4 = dRdt(Ii+dt*KI3, Ri+dt*KR3, d, gama, sigma)
    
    KS = (KS1 + 2*KS2 + 2*KS3 + KS4)/6.
    KI = (KI1 + 2*KI2 + 2*KI3 + KI4)/6.
    KR = (KR1 + 2*KR2 + 2*KR3 + KR4)/6.
    
    Si1 = Si + dt*KS
    Ii1 = Ii + dt*KI
    Ri1 = Ri + dt*KR
    Ni1 = Si1+Ii1+Ri1
    
    return np.array([Si1,Ii1,Ri1,Ni1])

#======Para SI e SIS
def dSdt_ (S,I,b,beta,d):
    N = 1.0*S+I
    return b*N - beta*S*I/N - d*S + gama*I

def dIdt_ (S,I,beta,d,delta,gama):
    N = 1.0*S+I
    return beta*S*I/N - d*I - delta*I - gama*I

def RK4_ (dt, Si, Ii, b, beta, d, delta, gama):
    
    KS1 = dSdt_(Si, Ii, b, beta, d)
    KI1 = dIdt_(Si, Ii, beta, d, delta, gama)
    
    KS2 = dSdt_(Si+dt*KS1/2, Ii+dt*KI1/2, b, beta, d)
    KI2 = dIdt_(Si+dt*KS1/2, Ii+dt*KI1/2, beta, d, delta, gama)
    
    KS3 = dSdt_(Si+dt*KS2/2, Ii+dt*KI2/2, b, beta, d)
    KI3 = dIdt_(Si+dt*KS2/2, Ii+dt*KI2/2, beta, d, delta, gama)
    
    KS4 = dSdt_(Si+dt*KS3, Ii+dt*KI3, b, beta, d)
    KI4 = dIdt_(Si+dt*KS3, Ii+dt*KI3, beta, d, delta, gama)
    
    KS = (KS1 + 2*KS2 + 2*KS3 + KS4)/6.
    KI = (KI1 + 2*KI2 + 2*KI3 + KI4)/6.
    
    Si1 = Si + dt*KS
    Ii1 = Ii + dt*KI
    Ni1 = Si1+Ii1

    return np.array([Si1,Ii1,Ni1])

st.title('MODELOS EPIDEMIOLÓGICOS')

modelo = st.sidebar.selectbox('Qual modelo epidemiológico deseja simular?','SI SIS SIR SIRS'.split())

if modelo == 'SI':
        #coletar as condições iniciais
    N0 = st.sidebar.number_input(
    'População', 
    min_value = 1000, 
    max_value = 1000000000, 
    value = 200000, 
    step = 1000)

    I0 = st.sidebar.number_input(
    'N° de Infectados', 
    min_value = 1, 
    max_value = N0, 
    value = 10, 
    step = 1)

    S0 = N0 - I0

    #variáveis
    beta = st.sidebar.slider(
    'Força da Infecção',
    0.0, 1.0, 0.01)

    b = st.sidebar.number_input(
    'Taxa de Natalidade (ao ano por mil habt)', 
    min_value = 0., 
    max_value = 25., 
    value = 14.0, 
    step = 0.1)/(1000*365)

    d = st.sidebar.number_input(
    'Taxa de Mortalidade (ao ano por mil habt)', 
    min_value = 0., 
    max_value = 25., 
    value = 6.0, 
    step = 0.1)/(1000*365)

    t_morte = st.sidebar.number_input(
    'Tempo para Morte (dias)', 
    min_value = 1, 
    max_value = 30, 
    value = 18, 
    step = 1)

    let = st.sidebar.number_input(
    'Letalidade da Doença (%)', 
    min_value = 0., 
    max_value = 100., 
    value = 1., 
    step = 0.1)

    # taxa de mortalidade da doença
    delta = (1./t_morte)*(let/100)

    # taxa de recuperação da doença
    #*** Para SI gama = 0
    gama = 0

    # por quantos dias rodar a simulação
    t_max = 720
    # taxa de variação do tempo
    dt = 1.

    #======================================================
    #### método de Runge Kutta de ordem 4 (RK4) ####
    #======================================================

    # Criar listas onde as informações vão ser adicionadas
    Tt4 = np.arange(t_max)*0.
    St4 = np.arange(t_max)*0.
    It4 = np.arange(t_max)*0.
    Nt4 = np.arange(t_max)*0.
    # add valores iniciais
    Tt4[0] = 0.
    St4[0] = S0
    It4[0] = I0
    Nt4[0] = S0+I0

    t = dt
    i = 1
    #Criar laço de repetição
    while t < t_max:
        
        Tt4[i] = t
        
        Sti = St4[i-1]
        Iti = It4[i-1]
        
        # Aplicar o RK4
        SI = RK4_(dt, Sti, Iti, b, beta, d, delta, gama)
        
        St4[i] = SI[0]
        It4[i] = SI[1]
        Nt4[i] = SI[2]
        
        t += dt
        i += 1 
elif modelo == 'SIS':
    #coletar as condições iniciais
    N0 = st.sidebar.number_input(
    'População', 
    min_value = 1000, 
    max_value = 1000000000, 
    value = 200000, 
    step = 1000)

    I0 = st.sidebar.number_input(
    'N° de Infectados', 
    min_value = 1, 
    max_value = N0, 
    value = 10, 
    step = 1)

    S0 = N0 - I0

    #variáveis
    beta = st.sidebar.slider(
    'Força da Infecção',
    0.0, 1.0, 0.2)

    b = st.sidebar.number_input(
    'Taxa de Natalidade (ao ano por mil habt)', 
    min_value = 0., 
    max_value = 25., 
    value = 14.0, 
    step = 0.1)/(1000*365)

    d = st.sidebar.number_input(
    'Taxa de Mortalidade (ao ano por mil habt)', 
    min_value = 0., 
    max_value = 25., 
    value = 6.0, 
    step = 0.1)/(1000*365)

    t_recup = st.sidebar.number_input(
    'Tempo de Recuperação (dias)', 
    min_value = 1, 
    max_value = 30, 
    value = 10, 
    step = 1)

    t_morte = st.sidebar.number_input(
    'Tempo para Morte (dias)', 
    min_value = 1, 
    max_value = 30, 
    value = 18, 
    step = 1)

    let = st.sidebar.number_input(
    'Letalidade da Doença (%)', 
    min_value = 0., 
    max_value = 100., 
    value = 1.5, 
    step = 0.1)

    # taxa de mortalidade da doença
    delta = (1./t_morte)*(let/100)

    # taxa de recuperação da doença
    gama = (1./t_recup)*((100-let)/100)

    # por quantos dias rodar a simulação
    t_max = 720
    # taxa de variação do tempo
    dt = 1.

    #======================================================
    #### método de Runge Kutta de ordem 4 (RK4) ####
    #======================================================

    # Criar listas onde as informações vão ser adicionadas
    Tt4 = np.arange(t_max)*0.
    St4 = np.arange(t_max)*0.
    It4 = np.arange(t_max)*0.
    Nt4 = np.arange(t_max)*0.
    # add valores iniciais
    Tt4[0] = 0.
    St4[0] = S0
    It4[0] = I0
    Nt4[0] = S0+I0

    t = dt
    i = 1
    #Criar laço de repetição
    while t < t_max:
        
        Tt4[i] = t
        
        Sti = St4[i-1]
        Iti = It4[i-1]
        
        # Aplicar o RK4
        SIS = RK4_ (dt, Sti, Iti, b, beta, d, delta, gama)
        
        St4[i] = SIS[0]
        It4[i] = SIS[1]
        Nt4[i] = SIS[2]
        
        t += dt
        i += 1 
elif modelo == 'SIR':
    #coletar as condições iniciais
    N0 = st.sidebar.number_input(
    'População', 
    min_value = 1000, 
    max_value = 1000000000, 
    value = 200000, 
    step = 1000)

    I0 = st.sidebar.number_input(
    'N° de Infectados', 
    min_value = 1, 
    max_value = N0, 
    value = 10, 
    step = 1)

    R0 = st.sidebar.number_input(
    'N° de Imunes', 
    min_value = 0, 
    max_value = N0 - I0, 
    value = 0, 
    step = 1)

    S0 = N0 - I0 - R0

    #variáveis
    beta = st.sidebar.slider(
    'Força da Infecção',
    0.0, 1.0, 0.2)

    b = st.sidebar.number_input(
    'Taxa de Natalidade (ao ano por mil habt)', 
    min_value = 0., 
    max_value = 25., 
    value = 14.0, 
    step = 0.1)/(1000*365)

    d = st.sidebar.number_input(
    'Taxa de Mortalidade (ao ano por mil habt)', 
    min_value = 0., 
    max_value = 25., 
    value = 6.0, 
    step = 0.1)/(1000*365)

    t_recup = st.sidebar.number_input(
    'Tempo de Recuperação (dias)', 
    min_value = 1, 
    max_value = 30, 
    value = 10, 
    step = 1)

    t_morte = st.sidebar.number_input(
    'Tempo para Morte (dias)', 
    min_value = 1, 
    max_value = 30, 
    value = 18, 
    step = 1)

    let = st.sidebar.number_input(
    'Letalidade da Doença (%)', 
    min_value = 0., 
    max_value = 100., 
    value = 1.5, 
    step = 0.1)

    # taxa de mortalidade da doença
    delta = (1./t_morte)*(let/100)

    # taxa de recuperação da doença
    gama = (1./t_recup)*((100-let)/100)

    # taxa com que as pessoas deixam de ser imunes
    #*** Para o SIR sigma = 0 
    sigma = 0

    # por quantos dias rodar a simulação
    t_max = 720
    # taxa de variação do tempo
    dt = 1.

    #======================================================
    #### método de Runge Kutta de ordem 4 (RK4) ####
    #======================================================

    # Criar listas onde as informações vão ser adicionadas
    Tt4 = np.arange(t_max)*0.
    St4 = np.arange(t_max)*0.
    It4 = np.arange(t_max)*0.
    Rt4 = np.arange(t_max)*0.
    Nt4 = np.arange(t_max)*0.
    # add valores iniciais
    Tt4[0] = 0.
    St4[0] = S0
    It4[0] = I0
    Rt4[0] = R0
    Nt4[0] = S0+I0+R0

    t = dt
    i = 1
    #Criar laço de repetição
    while t < t_max:
        
        Tt4[i] = t
        
        Sti = St4[i-1]
        Iti = It4[i-1]
        Rti = Rt4[i-1]
        
        # Aplicar o RK4
        SIR = RK4(dt, Sti, Iti, Rti, b, beta, d, delta, sigma, gama)
        
        St4[i] = SIR[0]
        It4[i] = SIR[1]
        Rt4[i] = SIR[2]
        Nt4[i] = SIR[3]
        
        t += dt
        i += 1 

elif modelo == 'SIRS':
    #coletar as condições iniciais
    N0 = st.sidebar.number_input(
    'População', 
    min_value = 1000, 
    max_value = 1000000000, 
    value = 200000, 
    step = 1000)

    I0 = st.sidebar.number_input(
    'N° de Infectados', 
    min_value = 1, 
    max_value = N0, 
    value = 10, 
    step = 1)

    R0 = st.sidebar.number_input(
    'N° de Imunes', 
    min_value = 0, 
    max_value = N0 - I0, 
    value = 0, 
    step = 1)

    S0 = N0 - I0 - R0

    #variáveis
    beta = st.sidebar.slider(
    'Força da Infecção',
    0.0, 1.0, 0.2)

    b = st.sidebar.number_input(
    'Taxa de Natalidade (ao ano por mil habt)', 
    min_value = 0., 
    max_value = 25., 
    value = 14.0, 
    step = 0.1)/(1000*365)

    d = st.sidebar.number_input(
    'Taxa de Mortalidade (ao ano por mil habt)', 
    min_value = 0., 
    max_value = 25., 
    value = 6.0, 
    step = 0.1)/(1000*365)

    t_recup = st.sidebar.number_input(
    'Tempo de Recuperação (dias)', 
    min_value = 1, 
    max_value = 30, 
    value = 10, 
    step = 1)

    t_morte = st.sidebar.number_input(
    'Tempo para Morte (dias)', 
    min_value = 1, 
    max_value = 30, 
    value = 18, 
    step = 1)

    let = st.sidebar.number_input(
    'Letalidade da Doença (%)', 
    min_value = 0., 
    max_value = 100., 
    value = 1.5, 
    step = 0.1)

    t_imuni = st.sidebar.number_input(
    'Duração da Imunidade (dias)', 
    min_value = 1, 
    max_value = 360, 
    value = 180, 
    step = 1)

    # taxa de mortalidade da doença
    delta = (1./t_morte)*(let/100)

    # taxa de recuperação da doença
    gama = (1./t_recup)*((100-let)/100)

    # taxa com que as pessoas deixam de ser imunes
    sigma = 1./t_imuni

    # por quantos dias rodar a simulação
    t_max = 720
    # taxa de variação do tempo
    dt = 1.

    #======================================================
    #### método de Runge Kutta de ordem 4 (RK4) ####
    #======================================================

    # Criar listas onde as informações vão ser adicionadas
    Tt4 = np.arange(t_max)*0.
    St4 = np.arange(t_max)*0.
    It4 = np.arange(t_max)*0.
    Rt4 = np.arange(t_max)*0.
    Nt4 = np.arange(t_max)*0.
    # add valores iniciais
    Tt4[0] = 0.
    St4[0] = S0
    It4[0] = I0
    Rt4[0] = R0
    Nt4[0] = S0+I0+R0

    t = dt
    i = 1
    #Criar laço de repetição
    while t < t_max:
        
        Tt4[i] = t
        
        Sti = St4[i-1]
        Iti = It4[i-1]
        Rti = Rt4[i-1]
        
        # Aplicar o RK4
        SIRS = RK4(dt, Sti, Iti, Rti, b, beta, d, delta, sigma, gama)
        
        St4[i] = SIRS[0]
        It4[i] = SIRS[1]
        Rt4[i] = SIRS[2]
        Nt4[i] = SIRS[3]
        
        t += dt
        i += 1 

if st.button('GRÁFICO'):
    ## Criar figura
    fig = go.Figure()

    # Adicionar gráficos a figura
    fig.add_trace(
        go.Scatter(x=Tt4,
               y=list(St4),
               name="Suscetível",
               line=dict(color="#024e73")))

    fig.add_trace(
        go.Scatter(x=Tt4,
                y=It4,
                name="Infectados",
                line=dict(color="#cc5800")))

    if modelo == 'SIR' or modelo == 'SIRS':
        fig.add_trace(
            go.Scatter(x=Tt4,
                    y=Rt4,
                    name="Recuperados",
                    line=dict(color="#008535")))

    fig.add_trace(
        go.Scatter(x=Tt4,
                y=Nt4,
                name="População",
                line=dict(color="#000000", dash='dash')))
    
    #Gravar Gráfico no app
    st.write(fig)
