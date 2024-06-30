#include <iostream>
#include <cstdio>
#include <cstring>
using namespace std;
#define ll long long 
#define fo(i,x,y) for(int i=(x);i<=(y);++i)
ll f[220][7][220];//n<=200,k<=6
ll n,k;
int main() {
    freopen("a.in","r",stdin);
    scanf("%lld%lld",&n,&k);
    memset(f,0,sizeof f);
    f[0][0][0]=1;
    fo(i,1,n) 
    {
        fo(j,1,i)
        {
            fo(l,1,k)
            {
                fo(jj,0,j)//jj<=j
                    f[i][l][j]+=f[i-j][l-1][jj];
            }
        }
    }
    ll ans = 0;
    fo(i,0,n) ans+=f[n][k][i];
    printf("%lld\n",ans);
}
// 64 位输出请用 printf("%lld")