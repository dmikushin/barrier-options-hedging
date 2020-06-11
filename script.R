#####################################################################
#                       Derivatives
#
# University: HEC Lausanne
# Programme: MScF
# Author: Dimitris Karyampas
# Date: 04.03.2020
#####################################################################


#####################################################################
#               Generate random paths for shares
#####################################################################

set.seed(123) # set the random seed https://en.wikipedia.org/wiki/Random_seed
nPaths<-10 # number of paths
nSteps<-252 # number of days
# dw<-matrix(nrow=nSteps, ncol=nPaths) # testing
# outer(1:nrow(dw), 1:ncol(dw), FUN=function(i,j) (i + j) / (nPaths + nSteps)) # testing
dw<-rnorm(nPaths*nSteps,0,1)
dw<-matrix(dw,nSteps,nPaths)
Spot<-13310
vola<-0.2
S<-Spot * exp(apply((-0.5*vola^2)*1/nSteps + vola*sqrt(1/nSteps)*dw,2,cumsum))
S<-rbind(rep(Spot,ncol(S)),S)

matplot(seq(1,253,1),S,type="l",lwd=2,xlab="Business days",ylab="Share price [$]",main="Random paths")
abline(v=seq(1,253,length.out = 15), col="lightgray", lty="dotted")
abline(h=seq(min(S),max(S),length.out = 15), col="lightgray", lty="dotted")
matpoints(seq(1,253,1),S,type="l",lwd=2)

#####################################################################
#       Generate correlated random paths for shares
#####################################################################

set.seed(123)
library(mvtnorm) # load library so to generate correlated random numbers with rmvnorm
sigma1<-0.3
sigma2<-0.3
rho<- 0.9
Sigma<- matrix(c(sigma1^2,rho*sigma1*sigma2,rho*sigma1*sigma2,sigma2^2), ncol=2)
dw<-rmvnorm(n = 252, mean = rep(0, nrow(Sigma)), sigma = Sigma,
            method=c("eigen", "svd", "chol"), pre0.9_9994 = FALSE)
Spot1<-13310
S1<-Spot1 * exp(cumsum((-0.5*sigma1^2)*1/252 + sqrt(1/252)*dw[,1]))
S1<-c(Spot1,S1)

Spot2<-1510
S2<-Spot2 * exp(cumsum((-0.5*sigma2^2)*1/252 + sqrt(1/252)*dw[,2]))
S2<-c(Spot2,S2)

ret1<-diff(log(S1))
ret2<-diff(log(S2))
plot(ret1,ret2,ylab="Underlying returns [%]",xlab="Future contract returns [%]",
     main="Log-return scatter")
abline(v=seq(-0.05,0.05,length.out = 15), col="lightgray", lty="dotted")
abline(h=seq(-0.05,0.05,length.out = 15), col="lightgray", lty="dotted")
points(ret1,ret2)
abline(a = 0, b = 1 * sign(rho), col=2, lwd=2)

#####################################################################
#                       Margining process
#####################################################################

par(mfrow=c(2,2))
set.seed(123)
nPaths<-10
nSteps<-252
dw<-rnorm(nPaths*nSteps,0,1)
dw<-matrix(dw,nSteps,nPaths)
Spot<-13310
vola<-0.2
S<-Spot * exp(apply((-0.5*vola^2)*1/252 + vola*sqrt(1/252)*dw,2,cumsum))
S<-rbind(rep(Spot,ncol(S)),S)
path<-3
plot(seq(1,nSteps+1,1),S[,path],type="l",lwd=2,xlab="Business days",ylab="Security payoff [$]",
     main="Future contract price")
abline(v=seq(1,nSteps+1,length.out = 15), col="lightgray", lty="dotted")
abline(h=seq(min(S),max(S),length.out = 15), col="lightgray", lty="dotted")
points(seq(1,nSteps+1,1),S[,path],type="l",lwd=2)

IM<-23634.27988 # initial margin (retrieved from Eurex website)
MM<-0.7*IM # define Maintenance margin as 70% of Initial margin
dailyPnL<-25*diff(S[,path])  # 25 is the contract multiplier
barplot(dailyPnL,main="25 x diff(FDAX)",ylab="Profit and Loss [$]", xlab="Business days")

marginAccountBalance<-marginAccountBalanceWithoutMC<-rep(0,length(dailyPnL))
marginCall<-rep(0,length(dailyPnL))
marginAccountBalance[1]<-IM
marginAccountBalanceWithoutMC[1]<-IM
for (i in 2:(length(dailyPnL)+1)){
        marginAccountBalanceWithoutMC[i]<-marginAccountBalanceWithoutMC[i-1] + dailyPnL[i-1] 
        marginAccountBalance[i]<-marginAccountBalance[i-1] + dailyPnL[i-1] + marginCall[i-1]
        marginCall[i]<- ifelse(marginAccountBalance[i] < MM, IM - marginAccountBalance[i], 0)
}
matplot(cbind(marginAccountBalance,marginAccountBalanceWithoutMC),type="l",col=1,
        main="Margin account balance [$]",ylab="Account balance [$]", xlab="Business days")
points(marginAccountBalance,type="l",col=4,lwd=2)
abline(h=IM,col=2,lwd=2)
barplot(cumsum(pmax(MM,marginAccountBalance) - marginAccountBalance),main="Cumulative margin call amount [$]", 
        xlab="Business days",ylab="Cumulative variation margin [$]")

