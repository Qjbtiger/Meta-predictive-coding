import numpy as np
import random
from numpy import log as ln

def f(x):
    return np.maximum(0,x)
    
def df(x):
    return np.where(x > 0, 1, 0)

def grad_clip(gra,bound):
    l2=(np.linalg.norm(gra,ord=2))
    if l2>=bound:
        return np.array((bound/l2)*gra)
    else:
        return np.array(gra)
def decay(epoch,ddl,lr0,step):
    lr = (0.5**(epoch//step))*lr0
    if lr<ddl:
        lr =ddl
    return lr
def turn1(mat):
    N=mat.shape[0]
    return(mat-np.diag(np.diag(mat))+np.eye(N)) #set the diagonal to be one
def turn0(mat):
    N=mat.shape[0]
    return(mat-np.diag(np.diag(mat))) #set the diagonal to be zero
def softmax(y):
    y = y - np.array(y.max(axis=0),ndmin=2)
    exp_y = np.exp(y) 
    sumofexp = np.array(exp_y.sum(axis=0),ndmin=2)
    softmax = exp_y/sumofexp
    return softmax
def sampling(m,p,v):
    import random
    m=np.array(m)
    p=np.array(p)
    v=np.array(v)
    b=np.ones((p.shape[0],p.shape[1]))
    ran=np.array(np.random.random((p.shape[0],p.shape[1]) ))
    b = (turn_2_zero_matrix(ran-p))*(np.random.normal(m,np.sqrt(v)))
    return b
def turn_2_zero_matrix(x):
    x1=x*1
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            if x[i][j]>0:
                x1[i][j]=1
            if x[i][j]<0:
                x1[i][j]=0
    return x1
def turn_2_zero(x):
    if x>0:
        return 1
    if x<=0:
        return 0

def randomdata_const(a,ep):
    data_set = np.zeros((26,100000))
    data_set[a][0] = 1
    loc=0
    value=a
    for i in range(ep):
        rep = 4
        value = np.random.choice([value+2,value+4,value+6,value+8],p = [0.25,0.25,0.25,0.25])
        up = 0
        down = 0
        if value>=26:
            value = value%26
        data_set[value][loc+1:loc+rep+1] = 1
        loc+=rep
    return data_set[:,0:loc].reshape(1,26,loc)

def full_dataset2(M,R,N):
  ##M:different possibilities that can go after a word，R：each letter repeats r times，N:round,sequence length is 1+NR
    data_al = []
    print("the expected length:",1+N*R)
    print("the expected load:",(M**(N))*26)
    for k in range(26):
        data_set = np.zeros((26,1+N*(R)))
        data_set[k,0] = 1 ##initialization
        data_all=[data_set]
        data_all2 = data_all
        value0 = k*1#initialization of value
        value = [k]
        value2=value*1
        for n in range(N):#round
            loc = 1+n*R #loc is deterministic
            data_all2 = []
            value2 = []
            for j in range(len(data_all)):
                value_temp = value[j]
                for m in range(M):#M possibilities, value+2, value+4, value+6, value+8
                    value_temp +=2
                    if value_temp>=26:
                        value_temp = value_temp%26
                    value2.append(value_temp*1)
                    data_temp = data_all[j]*1
                    data_temp[value_temp][loc:loc+R] = 1
                    data_all2.append((data_temp*1)*1)
            data_all=data_all2
            value = value2
        data_al.append(data_all*1)
    return data_al
                

#%% 训练和测试函数
 
def uni_permu(a1,b1,direction):
    a = a1
    b=b1
    if direction ==1:
        p = np.random.permutation(len(a.T))
        return np.array((a.T[p]).T), np.array((b.T[p]).T)
    if direction == 0:
        p = np.random.permutation(len(a))
        return np.array((a[p])), np.array((b[p]))
def decoding(x):
    #x 维度为 dimension * lenth
    code = ['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z']
    decode = []
    for i in range(x.shape[1]):
        decode.append(code[int(np.argmax(x[:,i]))]*1)
    return decode


# In[3]:


# M = 2
# R = 1
# N = 10
# Full_data = np.array(full_dataset2(M,R,N)).reshape(26*(M**(N)),26,1+N*R)
# train_data = Full_data[:,:,0:N*R]
# train_label = Full_data[:,:,1:N*R+1]


# In[84]:


def data_g(index):
    bs=len(index)
    sample = np.zeros((26,bs))
    for i in range(bs):
        sample[index[i]]=1
    return sample
class RNN:
    def __init__(self, n_in, n_rec, n_out, tau_m=10):
        self.n_in = n_in
        self.n_rec = n_rec
        self.n_out = n_out
        self.tau_m = tau_m

        # Initialization of parameters: pi, m, xi
        # 用于刻画权重w的分布
        # 后一层在第一维度
        np.random.seed()
        self.m_in =  np.array((np.random.normal(0,1,(n_rec,n_in)))/n_rec**0.5*n_in**0.5)*self.n_in
        self.xi_in = 0.1*np.random.random((n_rec,n_in))
        self.pi_in = np.random.uniform(0,0.5,(n_rec,n_in))
        
        self.m_rec = np.identity(n_rec)*self.n_rec
        self.xi_rec = 0.1*np.random.random((n_rec,n_rec))
        self.pi_rec = (np.random.uniform(0,0.5,(n_rec,n_rec)))
        
        self.m_out =  np.array((np.random.normal(0,1,(n_out,n_rec)))/n_rec**0.5*n_in**0.5)*self.n_rec
        self.xi_out = 0.1*np.random.random((n_out,n_rec))
        self.pi_out = np.random.uniform(0,0.5,(n_out,n_rec))
        self.mu_in,self.rho_in = self.mu_rho(self.n_in,self.m_in,self.pi_in,self.xi_in)
        self.mu_rec,self.rho_rec = self.mu_rho(self.n_rec,self.m_rec,self.pi_rec,self.xi_rec)
        self.mu_out,self.rho_out = self.mu_rho(self.n_rec,self.m_out,self.pi_out,self.xi_out)

        #adam 需要9个优化器
        self.Adam_m_in = Adam()
        self.Adam_xi_in = Adam()
        self.Adam_pi_in = Adam()
        self.Adam_m_rec = Adam()
        self.Adam_xi_rec = Adam()
        self.Adam_pi_rec = Adam()
        self.Adam_m_out = Adam()
        self.Adam_xi_out = Adam()
        self.Adam_pi_out = Adam() 

    def mu_rho(self,N,m,pi,xi):
        mu = m/N
        rho = m**2/(N**2*(1-pi))+xi/N
        return mu,rho
    def gra(self,N,m,pi,xi):
        mm = m*pi/(N*N*(1-pi))
        pp = m*m/(N*N*(1-pi)*(1-pi))
        xx = 1/N
        return mm,pp,xx
    def update_moment(self):
        self.mu_in,self.rho_in = self.mu_rho(self.n_in,self.m_in,self.pi_in,self.xi_in)
        self.mu_rec,self.rho_rec = self.mu_rho(self.n_rec,self.m_rec,self.pi_rec,self.xi_rec)
        self.mu_out,self.rho_out = self.mu_rho(self.n_rec,self.m_out,self.pi_out,self.xi_out)
    def variance_fix(self,delta):
        x = delta*1
        x[delta<0]=0
        return x
    def error(self,y,y_):
        return np.sum((y-y_)**2)/y.shape[2]

    def feedforward(self,hb,x,y_,f,m,fix=False):
        ##hb is belief
        t_max = np.shape(x)[0]#total time length  
        b_size = np.shape(x)[2]
        h = np.zeros((t_max, self.n_rec, b_size))##v
        yb = y_ #out dimension
        delta = np.zeros((t_max, self.n_rec, b_size))
        if fix == True:
            np.random.seed(m)
            epsi1=np.random.normal(0,1,(t_max,self.n_rec, b_size))
        else:
            epsi1=np.random.normal(0,1,(t_max,self.n_rec, b_size))
        delta_out = np.zeros((t_max, self.n_out, b_size))
        epsi2=np.random.normal(0,1,(t_max,self.n_out, b_size)) 
        y = np.zeros((t_max,self.n_out,b_size)) 
        out_ = np.zeros((t_max,self.n_out,b_size)) 
        err = np.zeros((t_max, self.n_rec, b_size))
        for tt in range(-1,t_max-1,1):
            self.update_moment()
            ##delta all not sqrt
            delta[tt+1] = np.maximum((self.rho_in - self.mu_in**2)@(x[tt+1]**2)+(self.rho_rec - self.mu_rec**2)@(f(hb[tt])**2),0)
            h[tt+1] = np.dot(self.mu_rec, f(hb[tt]))+ np.dot(self.mu_in, x[tt+1])+epsi1[tt+1]*np.sqrt(delta[tt+1])
            err[tt+1] = hb[tt+1]-h[tt+1]
            delta_out[tt+1] = np.maximum((self.rho_out - self.mu_out**2)@(f(hb[tt+1])**2),0)
            out_[tt+1] = np.dot(self.mu_out, f(hb[tt+1]))+epsi2[tt+1]*np.sqrt(delta_out[tt+1])
            y[tt+1] = softmax(out_[tt+1])
        err_out = y_-y
        return err,err_out,y,h,epsi1, epsi2,np.sqrt(delta),np.sqrt(delta_out)
    def feedhb(self,x,f,epsi1):
        t_max = np.shape(x)[0]#total time length  
        b_size = np.shape(x)[2]
        hb = 0*np.random.normal(0,1,(t_max+1, self.n_rec, b_size))##belief  
        hb[-1] =  0.0*np.ones([n_rec,b_size])
        delta = np.zeros((t_max, self.n_rec, b_size))
        for tt in range(-1,t_max-1,1):
            self.update_moment()
            delta[tt+1] = (np.maximum((self.rho_in - self.mu_in**2)@(x[tt+1]**2)+(self.rho_rec - self.mu_rec**2)@(f(hb[tt])**2),0))
            hb[tt+1] = np.dot(self.mu_rec, f(hb[tt]))+ np.dot(self.mu_in, x[tt+1])+epsi1[tt+1]*np.sqrt(delta[tt+1])
        return hb
    def feed2(self,x,y_,f):
        t_max = np.shape(x)[0]#total time length  
        b_size = np.shape(x)[2]
        h = np.zeros((t_max+1, self.n_rec, b_size))##v
        delta = np.zeros((t_max, self.n_rec, b_size))
        epsi1=np.random.normal(0,1,(t_max,self.n_rec, b_size))
        delta_out = np.zeros((t_max, self.n_out, b_size))
        epsi2=np.random.normal(0,1,(t_max,self.n_out, b_size)) 
        y = np.zeros((t_max,self.n_out,b_size)) 
        out_ = np.zeros((t_max,self.n_out,b_size)) 
        for tt in range(-1,t_max-1,1):
            self.update_moment()
            delta[tt+1] = np.maximum((self.rho_in - self.mu_in**2)@(x[tt+1]**2)+(self.rho_rec - self.mu_rec**2)@(f(h[tt])**2),0)
            h[tt+1] = np.dot(self.mu_rec, f(h[tt]))+ np.dot(self.mu_in, x[tt+1])+epsi1[tt+1]*np.sqrt(delta[tt+1])
            delta_out[tt+1] = np.maximum((self.rho_out - self.mu_out**2)@(f(h[tt+1])**2),0)
            out_[tt+1] = np.dot(self.mu_out, f(h[tt+1]))+epsi2[tt+1]*np.sqrt(delta_out[tt+1])
            y[tt+1] = softmax(out_[tt+1])
        return y
    def feed3(self,T,init,bs,error=False):
        t_max = T
        b_size = bs
        x=np.zeros((26,bs))
        x[init]=1
        h = np.zeros((t_max+1, self.n_rec, b_size))##v
        delta = np.zeros((t_max, self.n_rec, b_size))
        epsi1=np.random.normal(0,1,(t_max,self.n_rec, b_size))
        delta_out = np.zeros((t_max, self.n_out, b_size))
        epsi2=np.random.normal(0,1,(t_max,self.n_out, b_size)) 
        y = np.zeros((t_max,self.n_out,b_size)) 
        out_ = np.zeros((t_max,self.n_out,b_size)) 
        output = [x]
        for tt in range(-1,t_max-1,1):
            self.update_moment()
            delta[tt+1] = np.maximum((self.rho_in - self.mu_in**2)@(output[tt+1]**2)+(self.rho_rec - self.mu_rec**2)@(f(h[tt])**2),0)
            h[tt+1] = np.dot(self.mu_rec, f(h[tt]))+ np.dot(self.mu_in, output[tt+1])+epsi1[tt+1]*np.sqrt(delta[tt+1])
            delta_out[tt+1] = np.maximum((self.rho_out - self.mu_out**2)@(f(h[tt+1])**2),0)
            out_[tt+1] = np.dot(self.mu_out, f(h[tt+1]))+epsi2[tt+1]*np.sqrt(delta_out[tt+1])
            y[tt+1] = softmax(out_[tt+1])
            output.append(data_g(np.argmax(y[tt+1],axis=0).reshape(bs,1))*1)
        output = np.array(output)
        err_all2 = []
        num=0
        if error==False:
            return y
#             return decoding(np.squeeze(output).T)
        if error == True:
#             print(np.argmax(np.squeeze(output).T,axis=0))
            for i in range(T):
                if np.argmax(output[i,:,0]) == np.argmax(output[i+1,:,0])-2 or np.argmax(output[i,:,0]) == np.argmax(output[i+1,:,0])-4 or np.argmax(output[i,:,0]) == np.argmax(output[i+1,:,0])-2+26 or np.argmax(output[i,:,0]) == np.argmax(output[i+1,:,0])-4+26:
                    num+=1
            return num/T
    def test(self):
        error = []
        for i in range(26):
            sequence = self.feed3(10,i,1,error=True)
            error.append(sequence*1)
        return error
    def distribution(self,alp):
        for letter in range(26):
            x=np.zeros((26,1))
            x[letter]=1
            sequence = self.feed3(10,letter,1,error=False)
            if (np.argmax(sequence[0]))==letter+2 or (np.argmax(sequence[0]))==letter+4:
                print(letter)
                if max(sequence[0])<=1.0:
                    plt.figure(figsize=(6,4))
                    plt.tick_params(axis='both', labelsize=15)
                    plt.bar(range(len(sequence[0])), np.array(sequence[0]).reshape(26))
                    plt.xlabel("Letter index",fontsize=20)
                    plt.ylabel("Softmax output",fontsize=20)
                    plt.ylim(0,1)
                    plt.xticks(range(len(sequence[0])),['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z'])
                    plt.text(0,0.8,str(decoding(np.array(x))[0])+r'$\to$'+str(decoding(sequence[0])[0]),fontsize=25)
                    plt.savefig('./alpha'+str(alp)+'_letter'+str(letter)+'.pdf',bbox_inches = 'tight')
                    plt.show()
        
        
        
    def inference(self,x,y_,eta,f,df,epoch):
        self.update_moment()
        m=epoch
        t_max = np.shape(x)[0]#total time length  
        b_size = np.shape(x)[2]
        F=0
        hb = 0*np.random.normal(0,1,(t_max+1, self.n_rec, b_size))##belief  
        hb[-1] =  0.0*np.ones([n_rec,b_size])
        err,err_out,y,h,epsi1,epsi2,delta,delta_out = self.feedforward(hb,x,y_,f,m,fix =True)
        hb = self.feedhb(x,f,epsi1)
        err,err_out,y,h,epsi1,epsi2,delta,delta_out = self.feedforward(hb,x,y_,f,m,fix =True)
        num=100
        for i in range(num):
            self.update_moment()
            gx1 = np.zeros((t_max, self.n_rec, b_size))
            err,err_out,y,h,epsi1,epsi2,delta,delta_out = self.feedforward(hb,x,y_,f,epoch,fix=True)
            delta1 = pow(10,-30)*np.ones_like((delta))+delta
            delta2 = pow(10,-30)*np.ones_like((delta_out))+delta_out
            F1 = np.sum(0.5*(err**2))
            F2 = np.average((-y_*np.log(y+pow(10,-5))).sum(axis=0))
#             F2 = np.sum(0.5*(err_out**2))
#             print("err",F1)
#             print("err_out",F2)
            if F1>10000:
                break
            F=F1+F2
            for tt in range(t_max-1, -1, -1):
                self.update_moment()
                if tt==(t_max-1):
                    gx1[tt] = -eta*err[tt]+eta*df(hb[tt])*((self.mu_out.T)@err_out[tt])                    +eta*df(hb[tt])*f(hb[tt])*(((self.rho_out - self.mu_out**2).T)@(err_out[tt]*epsi2[tt]*self.variance_fix(delta_out[tt])/delta2[tt]))
                else:  
                    gx1[tt] = -eta*err[tt]+eta*df(hb[tt])*((self.mu_rec.T)@err[tt+1])                    +eta*f(hb[tt])*df(hb[tt])*(((self.rho_rec - self.mu_rec**2).T)@(err[tt+1]*epsi1[tt+1]*self.variance_fix(delta[tt+1])/(delta1[tt+1])))
                    +eta*df(hb[tt])*((self.mu_out.T)@err_out[tt])                    +eta*df(hb[tt])*f(hb[tt])*(((self.rho_out - self.mu_out**2).T)@(err_out[tt]*epsi2[tt]*self.variance_fix(delta_out[tt])/delta2[tt]))
                hb[tt]=hb[tt]+gx1[tt]
#         print("err",F1)
#         print("err_out",F2)
        return F,hb


    
    def parameter_updating(self, x, y_, alpha,epoch):
        self.update_moment()
        t_max = np.shape(x)[0]  
        b_size = np.shape(x)[2]
#         print("pi",np.sum(self.pi_rec))
#         print("xi",np.sum(self.xi_rec))
        F1,hb1 = self.inference(x,y_,0.1,f,df,epoch)
        gm_in=0
        gpi_in=0
        gxi_in=0
        gm_rec=0
        gpi_rec=0
        gxi_rec=0
        gm_out=0
        gpi_out=0
        gxi_out=0
        err,err_out,y,h,epsi1,epsi2,delta,delta_out = self.feedforward(hb1,x,y_,f,epoch)
        err1=self.error(y,y_)
        delta1 = pow(10,-30)*np.ones_like((delta))+delta
        delta_out1 = pow(10,-30)*np.ones_like((delta_out))+delta_out
        for tt in range(t_max):
            self.update_moment()
            gra_in = self.gra(self.n_in,self.m_in,self.pi_in,self.xi_in)
            data_in = (np.dot(err[tt]*epsi1[tt]/delta1[tt]*self.variance_fix(delta[tt]), (x[tt]**2).T)).reshape(self.n_rec,self.n_in)
            gm_in +=-err[tt]@(x[tt].T/self.n_in)-gra_in[0]*data_in
            gpi_in += -1/2*gra_in[1]*data_in
            gxi_in += -1/2*(gra_in[2]*data_in)          
            gra_rec = self.gra(self.n_rec,self.m_rec,self.pi_rec,self.xi_rec)
            h_rec = (np.dot(err[tt]*epsi1[tt]/delta1[tt]*self.variance_fix(delta[tt]),(f(hb1[tt-1])**2).T)).reshape(self.n_rec,self.n_rec)               
            gm_rec += -err[tt]@(f(hb1[tt-1]).T/self.n_rec)-gra_rec[0]*h_rec
            gpi_rec += -1/2*gra_rec[1]*h_rec
            gxi_rec += -1/2*gra_rec[2]*h_rec
            gra_out = self.gra(self.n_rec,self.m_out,self.pi_out,self.xi_out)
            h_out = (np.dot((err_out[tt])*epsi2[tt]/delta_out1[tt]*self.variance_fix(delta_out[tt]),(f(hb1[tt])**2).T)).reshape(self.n_out,self.n_rec)
            gm_out += -(err_out[tt])@(f(hb1[tt]).T/self.n_rec)-gra_out[0]*h_out
            gpi_out += -1/2*gra_out[1]*h_out
            gxi_out += -1/2*gra_out[2]*h_out
        bound = 10
        gm_in, gm_rec, gm_out = grad_clip(gm_in/b_size,bound), grad_clip(gm_rec/b_size,bound), grad_clip(gm_out/b_size,bound)
        gxi_in, gxi_rec, gxi_out = grad_clip(gxi_in/b_size,bound), grad_clip(gxi_rec/b_size,bound), grad_clip(gxi_out/b_size,bound)
        gpi_in, gpi_rec, gpi_out =grad_clip(gpi_in/b_size,bound), grad_clip(gpi_rec/b_size,bound), grad_clip(gpi_out/b_size,bound)

        self.m_in = self.Adam_m_in.New_theta(self.m_in,gm_in,alpha)
        self.pi_in = np.clip(self.Adam_pi_in.New_theta(self.pi_in,gpi_in,alpha),0,0.99)
        self.xi_in = (self.Adam_xi_in.New_theta(self.xi_in,gxi_in,alpha))
        
        self.m_rec = self.Adam_m_rec.New_theta(self.m_rec,gm_rec,alpha)
        self.pi_rec = np.clip(self.Adam_pi_rec.New_theta(self.pi_rec,gpi_rec,alpha),0,0.99)
        self.xi_rec = (self.Adam_xi_rec.New_theta(self.xi_rec,gxi_rec,alpha))
        
        
        self.m_out = self.Adam_m_out.New_theta(self.m_out,gm_out,alpha)
        self.pi_out = np.clip(self.Adam_pi_out.New_theta(self.pi_out,gpi_out,alpha),0,0.99)
        self.xi_out = (self.Adam_xi_out.New_theta(self.xi_out,gxi_out,alpha))
        self.xi_in = np.maximum(self.xi_in,0)
        self.xi_rec = np.maximum(self.xi_rec,0)
        self.xi_out = np.maximum(self.xi_out,0)
#         print("update1",np.sum(self.xi_rec))
#         print("update1",np.sum(self.pi_rec))
#         loss=np.sum(-y_*ln(y+pow(10,-30)))/b_size/10
#         print("acc1",self.error(y,y_))
#         print("acc2",self.error(self.feed2(x,y_,f),y_))
#         print("loss",loss)
#         print("perplexity",np.exp(loss))
        clr = 0
        for i in range(1):
            res=self.test()
            clr+=np.average(res)
            
#         if epoch%10==0:
#             print("try",self.feed3(10,0,1))
#             print("test",res)
#             print("test average",np.average(res))
        return F1,clr,y,y_
    def minibatch(self, b_size, batches, train_datax, train_labelx, learn_rate,epoch):
        self.update_moment()
        data,targets = uni_permu(train_datax,train_labelx,0)
        CEs = 0   
        F_all=[]
        acc=[]
        er=[]
        for ii in range(batches):
            self.update_moment()
            x = (data[ii*b_size:(ii+1)*b_size]).T
            y_ = (targets[ii*b_size:(ii+1)*b_size]).T
            F1,er1,y,y_ = self.parameter_updating(x, y_, learn_rate,epoch*batches+ii)
        return F1
    def SGD(self,epoch,b_size, batches, train_datax, train_labelx,start,every):
#         F1,hb1 = self.inference((data[0:500]).T,(targets[0:500]).T,0.1,f,df,epoch)
        error = []
#         print("initial",error)
        for i in range(epoch):
            self.update_moment()
            alpha = decay(i+1, 0.001,start,every)
#             print("test is",np.average(net1.xi_rec))
            F1 = self.minibatch(b_size, batches, train_datax, train_labelx, alpha,i)
#             print("test is",np.average(self.test()))
#             print("error",F1)
            error.append(F1*1)
        return error


class Adam:
    def __init__(self):
        self.lr=0.3
        self.beta1=0.9
        self.beta2=0.999
        self.epislon=1e-8
        self.m=0
        self.s=0
        self.t=0
    
    def initial(self):
        self.m = 0
        self.s = 0
        self.t = 0
    
    def New_theta(self,theta,gradient,eta):
        self.t += 1
        self.lr = eta
        self.decay=1e-4
        g=gradient
        self.m = self.beta1*self.m + (1-self.beta1)*g
        self.s = self.beta2*self.s + (1-self.beta2)*(g*g)
        self.mhat = self.m/(1-self.beta1**self.t)
        self.shat = self.s/(1-self.beta2**self.t)
        theta -= self.lr*((self.mhat/(pow(self.shat,0.5)+self.epislon))+self.decay*theta)
        return theta

