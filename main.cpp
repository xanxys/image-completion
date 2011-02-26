#include <set>
#include <iostream>
#include <fstream>
#include <vector>
#include <algorithm>
#include <functional>
#include <boost/bind.hpp>
#include <boost/foreach.hpp>
#include <boost/lambda/lambda.hpp>
#include <boost/filesystem/path.hpp>
#include <boost/filesystem/operations.hpp>
#include <boost/random/mersenne_twister.hpp>
#include <boost/graph/graph_traits.hpp>
#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/connected_components.hpp>
#include <opencv/cv.h>
#include <opencv/highgui.h>

using namespace cv;
using namespace std;
using namespace boost;
namespace fs=boost::filesystem;

#define SET_MEMBER(v,s) ((s).find(v)!=(s).end())

vector<int> random_subset(boost::mt19937& rng,int n,int k){
    while(true){
        set<int> s;
        for(int i=0;i<k;i++) s.insert(rng()%n);
        if(s.size()==k) return vector<int>(s.begin(),s.end());
    }
}

template<typename K>
bool comparing_values_in(vector<K>& vs,int i,int j){
    return vs[i]<vs[j];
}

const char* start_bold="\x1b[1;31;49m";
const char* end_bold="\x1b[0;0;0m";


const int dim_descriptor=288;

// produce gist-like descriptor of an image
// 16*16=256 edge descriptor
// 16*2=32 a,b
Mat image_descriptor(Mat& img){
    // rescale to enable descriptor comparison between images of different size
    const float mdim=400;
    
    Size sz=img.size();
    float r=(sz.width>sz.height)?(mdim/sz.width):(mdim/sz.height);
    
    Mat img_nrm;
    resize(img,img_nrm,Size(sz.width*r,sz.height*r),0,0,INTER_AREA);
    
    
    // separate channels into Lab
    Mat labc;
    cvtColor(img_nrm,labc,CV_RGB2Lab);
    
    Mat lab[3];
    split(labc,lab);
    
    
    // process L: calculate edge strength of 4*4*4*4 cells
    Mat kx=(Mat_<float>(1,3)<<1,-2,1);
    Mat ky=kx.t();
    Mat kp=0.25*(Mat_<float>(3,3)<<-1,0,1, 0,0,0, 1,0,-1);
    Mat km=-kp;
    Mat* kernels[4]={&kx,&ky,&kp,&km};
    
    float edges[4][4][4][4];
    
    Mat l=lab[0];
    for(int lv=0;lv<4;lv++){
        for(int ki=0;ki<4;ki++){
            Mat lx;
            filter2D(l,lx,CV_32F,*kernels[ki]);
            lx=max(lx,0);
            
            float w=lx.size().width;
            float h=lx.size().height;
            
            for(int i=0;i<4;i++){
                for(int j=0;j<4;j++){
                    Mat slx=lx(Range(0.25*j*h,0.25*(j+1)*h),Range(0.25*i*w,0.25*(i+1)*w));
                    float area=slx.size().width*slx.size().height;
                    
                    edges[lv][ki][i][j]=sum(slx)[0]/area;
                }
            }
        }
        
        Mat t;
        resize(l,t,Size(l.size().width/2,l.size().height/2));
        l=t;
    }
    
    
    // process a,b
    float as[4][4],bs[4][4];
    
    float w=lab[1].size().width;
    float h=lab[1].size().height;
    
    for(int i=0;i<4;i++){
        for(int j=0;j<4;j++){
            Mat sa=lab[1](Range(0.25*j*h,0.25*(j+1)*h),Range(0.25*i*w,0.25*(i+1)*w));
            Mat sb=lab[2](Range(0.25*j*h,0.25*(j+1)*h),Range(0.25*i*w,0.25*(i+1)*w));
            float area=w*h;
            
            as[i][j]=sum(sa)[0]/area;
            bs[i][j]=sum(sb)[0]/area;
        }
    }
    
    
    // pack them into 288 dimensional descriptor
    Mat m(288,1,CV_32F);
    
    for(int i=0;i<4;i++){
        for(int j=0;j<4;j++){
            int bix=i*4+j;
            int ofs=18*bix;
            
            for(int l=0;l<4;l++)
                for(int k=0;k<4;k++)
                    m.at<float>(ofs+l*4+k)=edges[l][k][i][j];
            
            m.at<float>(ofs+16)=as[i][j];
            m.at<float>(ofs+17)=bs[i][j];
        }
    }
    
    return m;
}



void write_descriptor_text(ostream& os,Mat& desc){
    for(int i=0;i<dim_descriptor;i++) os<<desc.at<float>(i)<<" ";
    os<<endl;
}

Mat read_descriptor_text(istream& is){
    Mat desc(dim_descriptor,1,CV_32F);
    for(int i=0;i<dim_descriptor;i++){
        float x;
        is>>x;
        desc.at<float>(i)=x;
    }
    
    return desc;
}

typedef vector<pair<Mat,string>> IndexGist;
typedef pair<Mat,Mat> IndexWeight;

void create_index_gist(string dir_path){
    cout<<start_bold<<"creating index.gist"<<end_bold<<endl;
    
    fs::path p(dir_path);
    
    const int min_dim=100; // reject images under this size
    
    int n=0;
    
    ofstream index("./index.gist",ios::out);
    
    fs::directory_iterator end;
    for(fs::directory_iterator it(p);it!=end;++it){
        cout<<"loading "<<*it<<" : ";
        Mat img=imread(it->string());
        
        if(img.size().width<min_dim || img.size().height<min_dim)
            cout<<"skipping"<<endl;
        else{
            cout<<img.size().width<<"*"<<img.size().height<<endl;
            
            Mat desc=image_descriptor(img);
            index<<*it<<endl;
            write_descriptor_text(index,desc);
            
            n++;
        }
    }
    
    index.close();
    
    cout<<n<<" images are indexed"<<endl;
}

IndexGist load_index_gist(){
    cout<<start_bold<<"loading index.gist"<<end_bold<<endl;

    ifstream index("./index.gist",ios::in);
    
    int n=0;
    vector<pair<Mat,string>> ix;
    while(!index.eof()){
        string path,desc;
        getline(index,path);
        getline(index,desc);
        
        if(path.size()==0 || desc.size()==0) break;
        
        istringstream ss(desc);
        Mat d=read_descriptor_text(ss);
        
        ix.push_back(pair<Mat,string>(d,path));
    }
    
    cout<<ix.size()<<" entries are loaded"<<endl;
    
    return ix;
}




void dump_index_gist(IndexGist& igist){
    cout<<start_bold<<"dumping index.gist"<<end_bold<<endl;
    
    cout<<"first 3dims before PCA"<<endl;
    ofstream naive("./naive.table",ios::out);
    for(int i=0;i<igist.size();i++){
        Mat& m=igist[i].first;
        naive<<m.at<float>(0)<<" "<<m.at<float>(1)<<" "<<m.at<float>(2)<<endl;
    }
    naive.close();
    
    
    cout<<"first 2dims after PCA"<<endl;
    
    Mat data(igist.size(),dim_descriptor,CV_32F);
    for(int i=0;i<igist.size();i++)
        for(int j=0;j<dim_descriptor;j++)
            data.row(i)=igist[i].first.t();
    
    PCA pca=PCA(data,Mat(),CV_PCA_DATA_AS_ROW);
    
    Mat data_good;
    pca.project(data,data_good);
    
    
    ofstream good("./PCA.table",ios::out);
    for(int i=0;i<igist.size();i++){
        Mat m=data_good.row(i);
        good<<m.at<float>(0)<<" "<<m.at<float>(1)<<" "<<m.at<float>(2)<<endl;
    }
    good.close();
}

void create_index_weight(IndexGist& igist){
}

IndexWeight load_index_weight(){
}



vector<string> search_image(vector<pair<Mat,string>>& index,Mat& query,int n){
    vector<float> ds;
    vector<int> is;
    
    ds.reserve(index.size());
    is.reserve(index.size());
    
    for(int i=0;i<index.size();i++){
        ds.push_back(norm(query,index[i].first));
        is.push_back(i);
    }
    sort(is.begin(),is.end(),bind(comparing_values_in<float>,ds,_1,_2)); 
   
    vector<string> ss;
    for(int i=0;i<n;i++)
        ss.push_back(index[is[i]].second);
    
    return ss;
}


// src must be defined in 1px boundary around mask
// grad_x(x,y):=f(x+1,y)-f(x,y), likewise for grad_y
void gradient_transfer(
    Mat& dst,Mat& mask,
    function<float(int,int)> src,
    function<float(int,int)> grad_x,
    function<float(int,int)> grad_y){
    
    const int n=countNonZero(mask);
    const int size[]={n,n};
    
    const int w=dst.size().width;
    const int h=dst.size().height;
    
    // constraint
    Mat sl(h,w,CV_32F),k(h,w,CV_32F);
    
    for(int y=0;y<h;y++){
        for(int x=0;x<w;x++){
            if(mask.at<uint8_t>(y,x)==0) continue;
            
            int ys[]={y-1,y,y,y+1};
            int xs[]={x,x-1,x+1,x};
            
            int ns=0;
            float vs=0;
            for(int i=0;i<4;i++){
                int xc=xs[i],yc=ys[i];
                if(xc<0 || xc>=w || yc<0 || yc>=h) continue;
                if(mask.at<uint8_t>(yc,xc)==0) vs+=src(xc,yc);
                
                if(xc==x){
                    if(yc<y)
                        vs+=grad_y(x,yc);
                    else
                        vs-=grad_y(x,y);
                }
                else{
                    if(xc<x)
                        vs+=grad_x(xc,y);
                    else
                        vs-=grad_x(x,y);
                }
                
                ns++;
            }
            
            k.at<float>(y,x)=1.0/ns;
            sl.at<float>(y,x)=vs;
        }
    }
    
    // solve iteratively
    float err_prev=1e100;
    while(true){
        float omega=1.9;
        float err=0;
        
        for(int y=0;y<h;y++)
            for(int x=0;x<w;x++)
                if(mask.at<uint8_t>(y,x)>0){
                    float vs=sl.at<float>(y,x);
                    
                    if(x>=1 && mask.at<uint8_t>(y,x-1)>0)
                        vs+=dst.at<float>(y,x-1);
                    if(y>=1 && mask.at<uint8_t>(y-1,x)>0)
                        vs+=dst.at<float>(y-1,x);
                    if(x<w-1 && mask.at<uint8_t>(y,x+1)>0)
                        vs+=dst.at<float>(y,x+1);
                    if(y<h-1 && mask.at<uint8_t>(y+1,x)>0)
                        vs+=dst.at<float>(y+1,x);
                    
                    vs*=k.at<float>(y,x);
                    
                    // successive over-relaxation
                    float vs_old=dst.at<float>(y,x);
                    vs=(1-omega)*vs_old+omega*vs;
                    dst.at<float>(y,x)=vs;
                                        
                    err+=abs(vs_old-vs);
                }
        
        cout<<"iter: "<<err<<endl;
        if(err>err_prev && err<100) break;
        
        err_prev=err;
    }
    
}

float aux_ref(Mat& m,int x,int y){
    return m.at<float>(y,x);
}

float aux_gx(Mat& m,int x,int y){
    return m.at<float>(y,x+1)-m.at<float>(y,x);
}

float aux_gy(Mat& m,int x,int y){
    return m.at<float>(y+1,x)-m.at<float>(y,x);
}

void blend_images(Mat& img,Mat& mask,Mat& filler){
    // enlarge filler to cover img
    float a_i=1.0*img.size().width/img.size().height;
    float a_f=1.0*filler.size().width/filler.size().height;
    float s=1;
    
    if(a_i>a_f)
        s=1.0*img.size().width/filler.size().width;
    else
        s=1.0*img.size().height/filler.size().height;
    
    Mat m;
    resize(filler,m,Size(),s,s);
    
    // seamless cloning
    vector<Mat> is;
    vector<Mat> fs;
    split(img,is);
    split(m,fs);
    
    assert(is.size()==fs.size());
    for(int i=0;i<is.size();i++){
        Mat i_c,f_c;
        is[i].convertTo(i_c,CV_32F);
        fs[i].convertTo(f_c,CV_32F);
        
        gradient_transfer(
            i_c,mask,
            bind(aux_ref,i_c,_1,_2),
            bind(aux_gx,f_c,_1,_2),
            bind(aux_gy,f_c,_1,_2));
        
        i_c.convertTo(is[i],CV_8U);
    }
    
    merge(is,img);
}



// parse argument and pass configuration to the composition function
int main(int argc,char *argv[]){
    IndexGist index;
    
    // interactive shell
    while(true){
        cout<<">"<<flush;
        
        string command;
        getline(cin,command);
        
        if(command=="q"){
            cout<<"exiting"<<endl;
            break;
        }
        else if(command=="create-anew-please-really"){
            create_index_gist("/data/flickr-large");
            index=load_index_gist();
        }
        else if(command=="dump.gist"){
            dump_index_gist(index);
        }
        else if(command=="create.weight"){
            create_index_weight(index);
        }
        else if(command=="load"){
            index=load_index_gist();
        }
        else if(command=="blend"){
            /*
            string d,dm,s;
            cout<<"image w/ hole?"<<flush;
            getline(cin,d);
            cout<<"image mask?"<<flush;
            getline(cin,dm);
            cout<<"filler image?"<<flush;
            getline(cin,s);
            */
            string
                d="hill.jpg",
                dm="hill_mask.jpg",
                s="hill_dual.jpeg";
            
            Mat id=imread(d);
            Mat idm=imread(dm,0);
            Mat is=imread(s);
            
            blend_images(id,idm,is);
            
            imwrite("b-"+d,id);
        }
        else if(command=="query"){
            int n;
            string path,d;
            
            cout<<"query image path?"<<flush;
            getline(cin,path);
            cout<<"query size?"<<flush;
            cin>>n;
            getline(cin,d);
            
            cout<<"retrieving images"<<endl;
            Mat qi=imread(path);
            Mat q=image_descriptor(qi);
            vector<string> paths=search_image(index,q,n);
            
            int i=0;
            for(auto it=paths.begin();it!=paths.end();++it){
                ostringstream fn;
                fn<<"result-"<<i++<<".jpeg";
                
                cout<<":"<<*it<<" > "<<fn.str()<<endl;
                Mat img=imread(*it);
                imwrite(fn.str(),img);
            }
            
            
            
            
        }
        else
            cout<<"unknown command: "<<command<<endl;
    }
    
    return 0;
}

