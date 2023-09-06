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
def uni_permu(a1,b1,direction):
    a = a1
    b=b1
    if direction ==1:
        p = np.random.permutation(len(a.T))
        return np.array((a.T[p]).T), np.array((b.T[p]).T)
    if direction == 0:
        p = np.random.permutation(len(a))
        return np.array((a[p])), np.array((b[p]))
def decay(epoch,ddl,lr0,step):
    lr = (0.5**(epoch//step))*lr0
    lr[lr<=ddl] = ddl
    return lr
def turn1(mat):
    N=mat.shape[0]
    return(mat-np.diag(np.diag(mat))+np.eye(N)) #把对角线置1
def turn0(mat):
    N=mat.shape[0]
    return(mat-np.diag(np.diag(mat))) #把对角线置0
def softmax(y):
    y = y - np.array(y.max(axis=0),ndmin=2)
    exp_y = np.exp(y) 
    sumofexp = np.array(exp_y.sum(axis=0),ndmin=2)
    softmax = exp_y/sumofexp
    return softmax
import load2
mnist=(load2.load_mnist(one_hot=True))
train_data = mnist[0][0][0:60000]
train_label = mnist[0][1][0:60000]
test_data = mnist[1][0][:10000]
test_label = mnist[1][1][:10000]
print(np.shape(train_data))
print(np.shape(train_label))


# In[2]:


class RNN:
    def __init__(self, n_in, n_rec, n_out, tau_m=10):
        ##dimension:26*100*26
        self.n_in = n_in
        self.n_rec = n_rec
        self.n_out = n_out
        self.tau_m = tau_m

        # Initialization of parameters: pi, m, xi
        # 用于刻画权重w的分布
        # 后一层在第一维度
        self.m_in =  np.array((np.random.normal(0,1,(n_rec,n_in)))/n_rec**0.5*n_in**0.5)*self.n_in
        self.xi_in = 0.1*np.random.random((n_rec,n_in))
        self.pi_in = np.random.uniform(0,0.1,(n_rec,n_in))
        
        self.m_rec = np.identity(n_rec)*self.n_rec
        self.xi_rec = 0.1*np.random.random((n_rec,n_rec))
        self.pi_rec = (np.random.uniform(0,0.1,(n_rec,n_rec)))
        
        self.m_out =  np.array((np.random.normal(0,1,(n_out,n_rec)))/n_rec**0.5*n_in**0.5)*self.n_rec
        self.xi_out = 0.1*np.random.random((n_out,n_rec))
        self.pi_out = np.random.uniform(0,0.1,(n_out,n_rec))
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
    def accuracy(self,y,y_):
        targets = y_.argmax(axis=0)
        predicts = y.argmax(axis=0)
        return np.sum(targets==predicts)/np.size(targets)*100

    def w_feed(self,x,y_,f,fix=False):
        t_max = np.shape(x)[0]#total time length  
        b_size = np.shape(x)[2]
        h = np.zeros((t_max+1, self.n_rec, b_size))##v
        yb = y_ #out dimension
        delta = np.zeros((t_max, self.n_rec, b_size))
        if fix == True:
            np.random.seed(42)
            epsi1=np.random.normal(0,1,(t_max,self.n_rec, b_size))
        else:
            epsi1=np.random.normal(0,1,(t_max,self.n_rec, b_size))
        delta_out = np.zeros((self.n_out, b_size))
        epsi2=np.random.normal(0,1,(self.n_out, b_size)) 
        y = np.zeros((self.n_out,b_size)) 
        out_ = np.zeros((self.n_out,b_size)) 
        for tt in range(-1,t_max-1,1):
            self.update_moment()
            delta[tt+1] = np.maximum((self.rho_in - self.mu_in**2)@(x[tt+1]**2)+(self.rho_rec - self.mu_rec**2)@(f(h[tt])**2),0)
            h[tt+1] = np.dot(self.mu_rec, f(h[tt]))+ np.dot(self.mu_in, x[tt+1])+epsi1[tt+1]*np.sqrt(delta[tt+1])
        delta_out = np.maximum((self.rho_out - self.mu_out**2)@(f(h[t_max-1])**2),0)
        out_ = np.dot(self.mu_out, f(h[t_max-1]))+epsi2*np.sqrt(delta_out)
        y = softmax(out_)
        return y,h

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
        delta_out = np.zeros((self.n_out, b_size))
        epsi2=np.random.normal(0,1,(self.n_out, b_size)) 
        y = np.zeros((self.n_out,b_size)) 
        out_ = np.zeros((self.n_out,b_size)) 
        err = np.zeros((t_max, self.n_rec, b_size))
        for tt in range(-1,t_max-1,1):
            self.update_moment()
            delta[tt+1] = np.maximum((self.rho_in - self.mu_in**2)@(x[tt+1]**2)+(self.rho_rec - self.mu_rec**2)@(f(hb[tt])**2),0)
            h[tt+1] = np.dot(self.mu_rec, f(hb[tt]))+ np.dot(self.mu_in, x[tt+1])+epsi1[tt+1]*np.sqrt(delta[tt+1])
            err[tt+1] = hb[tt+1]-h[tt+1]
        delta_out = np.maximum((self.rho_out - self.mu_out**2)@(f(hb[t_max-1])**2),0)
        out_ = np.dot(self.mu_out, f(hb[t_max-1]))+epsi2*np.sqrt(delta_out)
        y = softmax(out_)
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
        if epoch<40:
            num=100
        else:
            num=200
        for i in range(num):
            self.update_moment()
            gx1 = np.zeros((t_max, self.n_rec, b_size))
            err,err_out,y,h,epsi1,epsi2,delta,delta_out = self.feedforward(hb,x,y_,f,m,fix =True)
            delta1 = pow(10,-30)*np.ones_like((delta))+delta
            delta2 = pow(10,-30)*np.ones_like((delta_out))+delta_out
            F1 = np.sum(0.5*(err**2))
            F2 = np.average((-y_*np.log(y+pow(10,-5))).sum(axis=0))
#             F2 = np.sum(0.5*(err_out**2))
            if F1>10000:
                break
            F=F1+F2
            for tt in range(t_max-1, -1, -1):
                self.update_moment()
                if tt==(t_max-1):
                    gx1[tt] = -eta*err[tt]+eta*df(hb[tt])*((self.mu_out.T)@err_out)                    +eta*df(hb[tt])*f(hb[tt])*(((self.rho_out - self.mu_out**2).T)@(err_out*epsi2*self.variance_fix(delta_out)/delta2))
                else:  
                    gx1[tt] = -eta*err[tt]+eta*df(hb[tt])*((self.mu_rec.T)@err[tt+1])                    +eta*df(hb[tt])*f(hb[tt])*(((self.rho_rec - self.mu_rec**2).T)@(err[tt+1]*epsi1[tt+1]*self.variance_fix(delta[tt+1])/delta1[tt+1]))
                hb[tt]=hb[tt]+gx1[tt]
        print("err",F1)
        print("err_out",F2)
        return F,hb






    
    def parameter_updating(self, x, y_, alpha,epoch):
        t_max = np.shape(x)[0]  
        b_size = np.shape(x)[2]
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
#         err,err_out,y,h = self.feedforward(hb1,x,y_,f)
        err,err_out,y,h,epsi1,epsi2,delta,delta_out = self.feedforward(hb1,x,y_,f,epoch)
        delta1 = pow(10,-30)*np.ones_like((delta))+delta
        delta_out1 = pow(10,-30)*np.ones_like((delta_out))+delta_out
        for tt in range(t_max-1, -1, -1):
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
        h_out = (np.dot((err_out)*epsi2/delta_out1*self.variance_fix(delta_out),(f(hb1[t_max-1])**2).T)).reshape(self.n_out,self.n_rec)
        gm_out += -(err_out)@(f(hb1[t_max-1]).T/self.n_rec)-gra_out[0]*h_out
        gpi_out += -1/2*gra_out[1]*h_out
        gxi_out += -1/2*gra_out[2]*h_out
        
 
        gxi_in, gxi_rec, gxi_out = grad_clip(gxi_in/b_size,10), grad_clip(gxi_rec/b_size,10), grad_clip(gxi_out/b_size,10)
        gpi_in, gpi_rec, gpi_out =grad_clip(gpi_in/b_size,10), grad_clip(gpi_rec/b_size,10), grad_clip(gpi_out/b_size,10)

        self.m_in = self.Adam_m_in.New_theta(self.m_in,gm_in,alpha[0])
        self.pi_in = np.clip(self.Adam_pi_in.New_theta(self.pi_in,gpi_in,alpha[1]),0,0.99)
        self.xi_in = np.maximum(self.Adam_xi_in.New_theta(self.xi_in,gxi_in,alpha[2]),0)
        
        self.m_rec = self.Adam_m_rec.New_theta(self.m_rec,gm_rec,alpha[3])
        self.pi_rec = np.clip(self.Adam_pi_rec.New_theta(self.pi_rec,gpi_rec,alpha[4]),0,0.99)
        self.xi_rec = np.maximum(self.Adam_xi_rec.New_theta(self.xi_rec,gxi_rec,alpha[5]),0)
        
        
        self.m_out = self.Adam_m_out.New_theta(self.m_out,gm_out,alpha[6])
        self.pi_out = np.clip(self.Adam_pi_out.New_theta(self.pi_out,gpi_out,alpha[7]),0,0.99)
        self.xi_out = np.maximum(self.Adam_xi_out.New_theta(self.xi_out,gxi_out,alpha[8]),0)
        print("acc1",self.accuracy(y,y_))
        print("acc2",self.accuracy(self.w_feed(x,y_,f,fix=False)[0],y_))
        return F1
    

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


    
def train(net, b_size, batches, train_data, train_label, learn_rate,epoch):
    data,targets = uni_permu(train_data,train_label,0)
#     data,targets = data_shuffle(train_data,train_label)
    #计算均值
    CEs = 0   
    F_all=[]
    acc=[]
    er=[]
    for ii in range(batches):
        x = (data[ii*b_size:(ii+1)*b_size]).T.reshape(28,28,b_size)
        y_ = (targets[ii*b_size:(ii+1)*b_size]).T
        F1= net.parameter_updating(x, y_, learn_rate,epoch*batches+ii)      
        print("xirec",np.sum(net.xi_rec))
        print("xirec",np.sum(net.pi_rec))
        print('\rTraning process: {:.2f}%.'.format(100*(ii+1)/batches),end='')
    return F1,acc
def test(net, test_data, test_label,sampling=False):  
    accuracies = 0  
    batches = 10
    b_size = 1000
    data = test_data
    targets = test_label
    h_test = 0.1*np.ones([n_rec,b_size])
    for ii in range(batches):      
        x_test = (data[ii*b_size:(ii+1)*b_size].T).reshape(28,28,b_size)      
        y_test = (targets[ii*b_size:(ii+1)*b_size]).T
        if sampling:
            accuracy = net.sample_forward(x_test,y_test,h_test)
        else:
            accuracy=net.accuracy(net.w_feed(x_test,y_test,f,fix=False)[0],y_test)
        accuracies += accuracy
    print("xi_rec",np.sum(net.xi_rec**2))
    return accuracies/batches


batches = 200
b_size = 128
llr = pow(10,-2)
learn_rate = np.array([llr,llr,llr,llr,llr,llr,llr,llr,llr])
n_in, n_rec, n_out = 28, 100, 10
optimizer = 'Adam'
net1 = RNN(n_in, n_rec, n_out)
print('Training begin.')
data,targets = uni_permu(train_data,train_label,0)
F0,hb0 = net1.inference((data[0:500]).T.reshape(28,28,500),targets[0:500].T,0.1,f,df,0)
CEs = [F0*1]
accuracies = []
Total_epoch = 100
m_in_all=[]
m_out_all=[]
m_rec_all=[]
pi_in_all=[]
pi_out_all=[]
pi_rec_all=[]
sig_in_all=[]
sig_out_all=[]
sig_rec_all=[]        
print("initially",CEs)
if __name__ == '__main__':
	import time
	start = time.time()
	for epoch in range(Total_epoch):
		print('Epoch {}'.format(epoch))
	#     m_in_all.append(net1.m_in*1)
	#     m_out_all.append(net1.m_out*1)
	#     m_rec_all.append(net1.m_rec*1)
            
	#     pi_in_all.append(net1.pi_in*1)
	#     pi_out_all.append(net1.pi_out*1)
	#     pi_rec_all.append(net1.pi_rec*1)
            
	#     sig_in_all.append(net1.xi_in*1)
	#     sig_out_all.append(net1.xi_out*1)
	#     sig_rec_all.append(net1.xi_rec*1)    
    





		lr = decay(epoch+1, 1e-3, learn_rate, 40)
		CE = train(net1, b_size, batches, train_data, train_label, lr,epoch)    
    		CEs.append(CE[0]*1)
    		accuracy = test(net1, test_data, test_label,sampling=False)
   	 	accuracies.append(accuracy)
   		 #accuracy2=test(net1, test_data, test_label,sampling=True)
    		print(' Accuracy = {:.2f}%.'.format(accuracy))
    		print("F is",CEs)
    		#print(' CE = {:.4f}; Sampling Accuracy = {:.2f}%.'.format(CE,accuracy2*100))
		end = time.time()





