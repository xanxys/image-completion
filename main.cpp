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
#include <boost/graph/edge_list.hpp>
#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/graph_traits.hpp>
#include <boost/graph/graph_utility.hpp>
// #include <boost/graph/edmonds_karp_max_flow.hpp>
// #include <boost/graph/push_relabel_max_flow.hpp>
#include <boost/graph/kolmogorov_max_flow.hpp>
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

const char* begin_bold="\x1b[1;31;49m";
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
    cout<<begin_bold<<"creating index.gist"<<end_bold<<endl;
    
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
    cout<<begin_bold<<"loading index.gist"<<end_bold<<endl;

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
    cout<<begin_bold<<"dumping index.gist"<<end_bold<<endl;
    
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

vector<Mat> create_pyramid(Mat m,int n,float exponent=0){
    vector<Mat> pyr;
    
    for(int i=0;i<n;i++){
        pyr.push_back(m);
        
        Mat t;
        resize(m,t,Size(),0.5,0.5,INTER_LINEAR);

        m=t*pow(2,exponent);
    }
    
    return pyr;
}


// CV_32S
// s.size==t.size
// ex.size==ey.size is smaller than s.size by (1,1)
Mat optimize_seam(Mat& s,Mat& t,Mat& exf,Mat& eyf,Mat& exb,Mat& eyb){
    typedef adjacency_list_traits<vecS,vecS,directedS> Traits;
/*
    typedef adjacency_list<vecS,vecS,directedS,
        property<vertex_name_t,string,
            property<vertex_index_t,long>>,
        property<edge_capacity_t,long,
            property<edge_residual_capacity_t,long,
                property<edge_reverse_t,Traits::edge_descriptor>>>> Graph;
*/
  typedef adjacency_list < vecS, vecS, directedS,
    property < vertex_name_t, std::string,
    property < vertex_index_t, long,
    property < vertex_color_t, boost::default_color_type,
    property < vertex_distance_t, long,
    property < vertex_predecessor_t, Traits::edge_descriptor > > > > >,
    
    property < edge_capacity_t, long,
    property < edge_residual_capacity_t, long,
    property < edge_reverse_t, Traits::edge_descriptor > > > > Graph;
    
    const int w=s.size().width;
    const int h=s.size().height;
    
    // construct graph
    int vs=w*h;
    int vt=w*h+1;

    Graph g(w*h+2);
        
    property_map<Graph,edge_reverse_t>::type reverse_edge = get(edge_reverse, g);
    property_map<Graph,edge_residual_capacity_t>::type residual_capacity = get(edge_residual_capacity, g);
    property_map<Graph,edge_capacity_t>::type capacity=get(edge_capacity,g);

    for(int y=0;y<h;y++){
        for(int x=0;x<w;x++){
            typedef pair<pair<int,int>,pair<int,int>> EdgeInfo;
            vector<EdgeInfo> edges;
            
            int v=y*w+x;
            int vx=v+1;
            int vy=v+w;
            
            edges.push_back(EdgeInfo(pair<int,int>(vs,v),pair<int,int>(s.at<int32_t>(y,x),0)));
            edges.push_back(EdgeInfo(pair<int,int>(v,vt),pair<int,int>(t.at<int32_t>(y,x),0)));
            
            if(x<w-1)
                edges.push_back(EdgeInfo(pair<int,int>(v,vx),pair<int,int>(exf.at<int32_t>(y,x),exb.at<int32_t>(y,x))));
            if(y<h-1)
                edges.push_back(EdgeInfo(pair<int,int>(v,vy),pair<int,int>(eyf.at<int32_t>(y,x),eyb.at<int32_t>(y,x))));
            
            for(auto it=edges.begin();it!=edges.end();++it){
                int u=it->first.first;
                int v=it->first.second;
                int capf=it->second.first;
                int capb=it->second.second;
                
                Traits::edge_descriptor e1,e2;
                e1=add_edge(u,v,g).first;
                e2=add_edge(v,u,g).first;
                
                reverse_edge[e1]=e2;
                reverse_edge[e2]=e1;
                
                capacity[e1]=capf;
                capacity[e2]=capb;
                
                residual_capacity[e1]=capf;
                residual_capacity[e2]=capb;
            }
        }
    }

    
    cout<<"graph constructed"<<endl;
    
    long flow=kolmogorov_max_flow(g,vs,vt);
//    long flow=edmonds_karp_max_flow(g,vs,vt);
    cout<<"max flow:"<<flow<<endl;
    
    
    Mat mask(h,w,CV_8U);
    
    graph_traits < Graph >::out_edge_iterator ei,e_end;
    for(tie(ei, e_end) = out_edges(vs, g); ei != e_end; ++ei){
    //    cout<<"  RCap:"<<residual_capacity[*ei]<<"  Cap:"<<capacity[*ei]<<endl;
        
        int v=target(*ei,g);

        int x=v%w;
        int y=v/w;
        
        
        Traits::edge_descriptor en=edge(v,vt,g).first;
        
//        cout<<x<<","<<y<<": "<<residual_capacity[*ei]<<" "<<residual_capacity[en]<<endl;
        
        
//        cout<<residual_capacity[*ei]<<endl;

        if(residual_capacity[*ei]==0) // capacity[*ei])
            mask.at<uint8_t>(y,x)=255;
        else
            mask.at<uint8_t>(y,x)=0;

    }

    return mask;
}


// src must be defined in 1px boundary around mask
// grad_x(x,y):=f(x+1,y)-f(x,y), likewise for grad_y
void gradient_transfer(
    Mat& dst0,Mat& mask0,
    function<float(int,int)> src,
    function<float(int,int)> grad_x,
    function<float(int,int)> grad_y){
    
    const int w0=dst0.size().width;
    const int h0=dst0.size().height;
    
    // constraint
    Mat sl0(h0,w0,CV_32F),inv_diag0(h0,w0,CV_32F);
    
    for(int y=0;y<h0;y++){
        for(int x=0;x<w0;x++){
            if(mask0.at<uint8_t>(y,x)==0){
                inv_diag0.at<float>(y,x)=0;
                sl0.at<float>(y,x)=0;
            }
            else{
                int ys[]={y-1,y,y,y+1};
                int xs[]={x,x-1,x+1,x};
                
                int ns=0;
                float vs=0;
                for(int i=0;i<4;i++){
                    int xc=xs[i],yc=ys[i];
                    if(xc<0 || xc>=w0 || yc<0 || yc>=h0) continue;
                    if(mask0.at<uint8_t>(yc,xc)==0) vs+=src(xc,yc);
                    
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
                
                inv_diag0.at<float>(y,x)=1.0/ns;
                sl0.at<float>(y,x)=vs;
            }
        }
    }
    
    const bool true_solution=false;
    
    // true: Gauss-Seidel + over-relaxation (very slow)
    // non-true: multigrid (experimental, buggy, very fast)
    
    const int levels=true_solution?1:5;
    
    vector<Mat>
        pyr_mask=create_pyramid(mask0,levels),
        pyr_inv_diag=create_pyramid(inv_diag0,levels,0);
    
    vector<Mat> pyr_dst,pyr_res;
    
    // fine -> coarse
    Mat sl(sl0);
    Mat dst(dst0);
    
    for(int l=0;l<levels;l++){
        cout<<"level: "<<l<<endl;
        
        Mat& mask=pyr_mask[l];
        Mat& inv_diag=pyr_inv_diag[l];
        
        int w=dst.size().width;
        int h=dst.size().height;
        
        float err_prev=1e100;
        int n_iter=0;
        
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
                        
                        vs*=inv_diag.at<float>(y,x);
                        
                        // successive over-relaxation
                        float vs_old=dst.at<float>(y,x);
                        vs=(1-omega)*vs_old+omega*vs;
                        dst.at<float>(y,x)=vs;
                                            
                        err+=abs(vs_old-vs);
                    }
            
            cout<<"iter: "<<err<<endl;
            
            if(true_solution){
                if(err>err_prev && err<100) break;
            }
            else{
                if(n_iter++>100) break;
            }
            
            err_prev=err;
        }
        
        pyr_dst.push_back(dst);
        
        // residue calculation
        Mat res;
        dst.copyTo(res);
        
        for(int y=0;y<h;y++)
            for(int x=0;x<w;x++)
                if(mask.at<uint8_t>(y,x)>0){
                    float vs=dst.at<float>(y,x)/inv_diag.at<float>(y,x);
                    
                    if(x>=1 && mask.at<uint8_t>(y,x-1)>0)
                        vs-=dst.at<float>(y,x-1);
                    if(y>=1 && mask.at<uint8_t>(y-1,x)>0)
                        vs-=dst.at<float>(y-1,x);
                    if(x<w-1 && mask.at<uint8_t>(y,x+1)>0)
                        vs-=dst.at<float>(y,x+1);
                    if(y<h-1 && mask.at<uint8_t>(y+1,x)>0)
                        vs-=dst.at<float>(y+1,x);
                    

                    res.at<float>(y,x)=sl.at<float>(y,x)-vs;
                }
                else
                    res.at<float>(y,x)=0;
        
        pyr_res.push_back(res);
        
        // scaling
        if(l<levels-1){
            Mat res_;
            resize(res,res_,Size(),0.5,0.5);
            
            sl=res_;
            sl.copyTo(dst);
        }
            
        ostringstream os;
        os<<"aux-MG-lv-"<<l<<".jpeg";
        imwrite(os.str(),dst);

        imwrite("res-"+os.str(),0.1*res.size().area()*abs(res)/norm(res));
    }
    
    // coarse -> fine
    for(int l=levels-2;l>=0;l--){
        Mat d_;
        resize(pyr_dst[l+1],d_,pyr_dst[l].size());
        
        pyr_dst[l]+=d_*4;

        ostringstream os;
        os<<"dst-MG-lv-"<<l<<".jpeg";
        imwrite(os.str(),pyr_dst[l]);
    }
    
    pyr_dst[0].copyTo(dst);
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

float aux_gx_large(Mat& m,Mat& n,int x,int y){
    float gm=pow(aux_gx(m,x,y),2)+pow(aux_gy(m,x,y),2);
    float gn=pow(aux_gx(n,x,y),2)+pow(aux_gy(n,x,y),2);
    
    return aux_gx(gm>gn?m:n,x,y);
}

float aux_gy_large(Mat& m,Mat& n,int x,int y){
    float gm=pow(aux_gx(m,x,y),2)+pow(aux_gy(m,x,y),2);
    float gn=pow(aux_gx(n,x,y),2)+pow(aux_gy(n,x,y),2);
    
    return aux_gy(gm>gn?m:n,x,y);
}

void blend_images(Mat& img,Mat& mask,Mat& filler){
    cout<<begin_bold<<"blending images"<<end_bold<<endl;
    
    // enlarge filler to cover img
    float a_i=1.0*img.size().width/img.size().height;
    float a_f=1.0*filler.size().width/filler.size().height;
    
    Mat m;    
    if(a_i>a_f){
        float s=1.0*img.size().width/filler.size().width;
        resize(filler,m,Size(),s,s);
        
        float dh_half=(m.size().height-img.size().height)/2;
        m=Mat(m,Rect(Size(0,dh_half),img.size()));
    }
    else{
        float s=1.0*img.size().height/filler.size().height;
        resize(filler,m,Size(),s,s);
        
        float dv_half=(m.size().width-img.size().width)/2;
        m=Mat(m,Rect(Size(dv_half,0),img.size()));
    }
    
    assert(img.size()==m.size());
    
    const int w=img.size().width;
    const int h=img.size().height;
    
    
    // find optimal seam
    Mat s(h,w,CV_32S),
        t(h,w,CV_32S),
        exf(h,w-1,CV_32S),
        exb(h,w-1,CV_32S),
        eyf(h-1,w,CV_32S),
        eyb(h-1,w,CV_32S);
    
    Mat inv_mask=255-mask;
    
    Mat ext_mask;
    dilate(mask,ext_mask,Mat());
    ext_mask=ext_mask<=mask;
    
    Mat dist,dist_bnd;
    distanceTransform(inv_mask,dist,CV_DIST_L2,3);
    distanceTransform(ext_mask,dist_bnd,CV_DIST_L2,3);
    
    imwrite("aux-dist0.jpeg",dist);
    imwrite("aux-dist1.jpeg",dist_bnd);
    
    Mat imgfx,mfx;
    cvtColor(img,imgfx,CV_RGB2Lab);
    cvtColor(m,mfx,CV_RGB2Lab);
    
    Mat imgf,mf;
    imgfx.convertTo(imgf,CV_32F);
    mfx.convertTo(mf,CV_32F);
    
    
    float we=5.0;
    
    for(int y=0;y<h;y++)
        for(int x=0;x<w;x++){
            if(mask.at<uint8_t>(y,x)>0)
                t.at<int32_t>(y,x)=0x7fffffff;
            else
                t.at<int32_t>(y,x)=0;
            
            s.at<int32_t>(y,x)=0.2*pow(dist.at<float>(y,x),2);            
            
            if(x<w-1){
                float f0=norm(Mat(imgf.at<Vec3f>(y,x)),Mat(mf.at<Vec3f>(y,x)));
                float fx=norm(Mat(imgf.at<Vec3f>(y,x+1)),Mat(mf.at<Vec3f>(y,x+1)));
                
                float df=norm(Mat(imgf.at<Vec3f>(y,x)),Mat(mf.at<Vec3f>(y,x+1)));
                float db=norm(Mat(mf.at<Vec3f>(y,x)),Mat(imgf.at<Vec3f>(y,x+1)));
                
                exf.at<int32_t>(y,x)=we*abs(fx-f0);
                exb.at<int32_t>(y,x)=we*abs(fx-f0);
            }
            if(y<h-1){
                float f0=norm(Mat(imgf.at<Vec3f>(y,x)),Mat(mf.at<Vec3f>(y,x)));
                float fy=norm(Mat(imgf.at<Vec3f>(y+1,x)),Mat(mf.at<Vec3f>(y+1,x)));

                float df=norm(Mat(imgf.at<Vec3f>(y,x)),Mat(mf.at<Vec3f>(y+1,x)));
                float db=norm(Mat(mf.at<Vec3f>(y,x)),Mat(imgf.at<Vec3f>(y+1,x)));
                
                eyf.at<int32_t>(y,x)=we*abs(fy-f0);
                eyb.at<int32_t>(y,x)=we*abs(fy-f0);
            }
        }
    
    Mat seam=optimize_seam(s,t,exf,eyf,exb,eyb);
    
    Mat xxxl;
    vector<Mat> chs;
    chs.push_back(mask);
    chs.push_back(seam);
    chs.push_back(seam);
    merge(chs,xxxl);
    imwrite("aux-revised-mask.jpeg",xxxl);
    
    mask=seam;
    
    
    // seamless cloning
    const bool use_cloning=true;
    
    vector<Mat> is;
    vector<Mat> fs;
    split(img,is);
    split(m,fs);
    
    assert(is.size()==fs.size());
    for(int i=0;i<is.size();i++){
        Mat i_c,f_c;
        is[i].convertTo(i_c,CV_32F);
        fs[i].convertTo(f_c,CV_32F);
        
        if(use_cloning){
            gradient_transfer(
                i_c,mask,
                bind(aux_ref,i_c,_1,_2),
                bind(aux_gx,f_c,_1,_2),
                bind(aux_gy,f_c,_1,_2));
        }
        else{
            gradient_transfer(
                i_c,mask,
                bind(aux_ref,i_c,_1,_2),
                bind(aux_gx_large,f_c,i_c,_1,_2),
                bind(aux_gy_large,f_c,i_c,_1,_2));
        }
        
        i_c.convertTo(is[i],CV_8U);
    }
    
    merge(is,img);
}

void change_color(Mat& img,Mat& mask){
    Mat L;
    cvtColor(img,L,CV_RGB2GRAY);
    
    vector<Mat> is;
    split(img,is);
    
    for(int i=0;i<is.size();i++){
        Mat l_c,i_c;
        L.convertTo(l_c,CV_32F);
        is[i].convertTo(i_c,CV_32F);
        
        gradient_transfer(
            l_c,mask,
            bind(aux_ref,l_c,_1,_2),
            bind(aux_gx,i_c,_1,_2),
            bind(aux_gy,i_c,_1,_2));
        
        l_c.convertTo(is[i],CV_8U);
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
        else if(command=="fill"){
            int n;
            string d,dm,dummy;
            cout<<"image w/ hole?"<<flush;
            getline(cin,d);
            cout<<"image mask?"<<flush;
            getline(cin,dm);
            cout<<"#candidates?"<<flush;
            cin>>n;
            getline(cin,dummy);

            
            Mat id=imread(d);
            Mat idm=imread(dm,0);
            threshold(idm,idm,127,255,THRESH_BINARY);
            
            cout<<"retrieving images"<<endl;
            Mat qi=imread(d);
            Mat q=image_descriptor(qi);
            vector<string> paths=search_image(index,q,n);
            
            cout<<"composing images"<<endl;
            int i=0;
            for(auto it=paths.begin();it!=paths.end();++it){
                ostringstream fn;
                fn<<"result-"<<i++<<".jpeg";
                
                cout<<":"<<*it<<" > "<<fn.str()<<endl;
                
                Mat img=imread(*it);
                
                Mat dst,mask;
                id.copyTo(dst);
                idm.copyTo(mask);
                
                blend_images(dst,mask,img);
                imwrite(fn.str(),dst);
            }
        }
        else if(command=="blend"){
            string d,dm,s;
            cout<<"image w/ hole?"<<flush;
            getline(cin,d);
            cout<<"image mask?"<<flush;
            getline(cin,dm);
            cout<<"filler image?"<<flush;
            getline(cin,s);
            
            Mat id=imread(d);
            Mat idm=imread(dm,0);
            threshold(idm,idm,127,255,THRESH_BINARY);
            Mat is=imread(s);
            
            blend_images(id,idm,is);
            
            imwrite("b-"+d,id);
        }
        else if(command=="color.partial"){
            string d,dm;
            cout<<"original?"<<flush;
            getline(cin,d);
            cout<<"image mask?"<<flush;
            getline(cin,dm);
            
            Mat id=imread(d);
            Mat idm=imread(dm,0);
            threshold(idm,idm,127,255,THRESH_BINARY);
            
            change_color(id,idm);
            
            imwrite("cp-"+d,id);
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

