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

#define foreach BOOST_FOREACH
#define SET_MEMBER(v,s) ((s).find(v)!=(s).end())

vector<int> random_subset(boost::mt19937& rng,int n,int k){
    while(true){
        set<int> s;
        for(int i=0;i<k;i++) s.insert(rng()%n);
        if(s.size()==k) return vector<int>(s.begin(),s.end());
    }
}

const char* start_bold="\x1b[1;31;49m";
const char* end_bold="\x1b[0;0;0m";

// match two sets of descriptors and gives average feature distance
vector<pair<int,int>> match_descriptors(Mat& s,Mat& t,double& score){
    const int sn=s.size().height;
    const int tn=t.size().height;

    cout<<"matching: "<<sn<<" - "<<tn<<endl;
    
    // construct precursor of bijection s<->t
    vector<pair<int,int>> pairs;
    set<int> exist_ti,bad_ti;
    
    for(int si=0;si<sn;si++){
        // find minimum
        double md=1e100;
        int mi=-1;
        for(int ti=0;ti<tn;ti++){
            double d=norm(s.row(si),t.row(ti));
            if(d<md){
                md=d;
                mi=ti;
            }
        }
        
        if(SET_MEMBER(mi,exist_ti)) bad_ti.insert(mi);
        exist_ti.insert(mi);
        pairs.push_back(pair<int,int>(si,mi));
    }
    
    // delete pairs with duplicated t index
    vector<pair<int,int>> pairs_ok;
    for(vector<pair<int,int>>::iterator it=pairs.begin();it!=pairs.end();++it)
        if(!SET_MEMBER(it->second,bad_ti))
            pairs_ok.push_back(*it);
    
    // calculate score
    double sc=0;
    for(vector<pair<int,int>>::iterator it=pairs_ok.begin();it!=pairs_ok.end();++it)
        sc+=norm(s.row(it->first),t.row(it->second));
    
    score=sc/pairs_ok.size();
    
    
    cout<<">"<<pairs_ok.size()<<" pairs established / score="<<score<<endl;
    
    return pairs_ok;
}

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
            filter2D(l,lx,CV_32F,kx);
            lx=abs(lx);
            
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




void compose_directory(fs::path p,fs::path q){
    // load all images
    vector<Mat> imgs;
    
    fs::directory_iterator end;
    for(fs::directory_iterator it(p);it!=end;++it){
        cout<<"loading "<<*it<<" : ";
        Mat img=imread(it->string());
        cout<<img.size().width<<"*"<<img.size().height<<endl;
        imgs.push_back(img);
    }
    
    
    // detect features
    cout<<"detecting features..."<<endl;
    
    Ptr<FeatureDetector> detector=GoodFeaturesToTrackDetector::create("SURF"); // new SurfFeatureDetector(1e3);
    vector<vector<KeyPoint>> kpss;
    detector->detect(imgs,kpss);
    
    // calculate descriptors
    cout<<"calculating descriptors..."<<endl;
    
    Ptr<DescriptorExtractor> extractor=DescriptorExtractor::create("SURF");
    vector<Mat> ds;
    extractor->compute(imgs,kpss,ds);

    // match images / construct feature point graph
    cout<<"matching descriptors..."<<endl;
    
    typedef property<vertex_name_t,pair<int,int>> VertexProperty;
    typedef adjacency_list<vecS,vecS,undirectedS,VertexProperty> Graph;
    typedef graph_traits<Graph>::vertex_descriptor Vertex;
    
    const int n=imgs.size();
    if(n!=2) return;
    
    Graph g;
    
    // register vertices
    vector<vector<Vertex>> vss;
    for(int i=0;i<n;i++){
        vector<Vertex> vs;
        for(int j=0;j<kpss[i].size();j++)
            vs.push_back(add_vertex(pair<int,int>(i,j),g));
        vss.push_back(vs);
    }
    
    // register edges
    for(int i=0;i<n;i++){
        for(int j=i+1;j<n;j++){
            double dist;
            vector<pair<int,int>> ls=match_descriptors(ds[i],ds[j],dist);
            
            pair<int,int> p;
            foreach(p,ls){
                add_edge(vss[i][p.first],vss[j][p.second],g);
            }
        }
    }
    
    // compute CCs and filter them
    cout<<"constructing descriptor hypergraph..."<<endl;
    
    vector<int> comp(num_vertices(g));
    int ncc=connected_components(g,&comp[0]);
    
    cout<<">"<<ncc<<" CCs found"<<endl;
    
    vector<vector<pair<int,int>>> features(ncc);
    for(int i=0;i<num_vertices(g);i++){

        features[comp[i]].push_back(get(vertex_name,g,i));
    }
    
    vector<vector<pair<int,int>>> features_good;
    for(int i=0;i<ncc;i++){
        vector<pair<int,int>>& x=features[i];
        if(x.size()<2) continue; // isolated
        
        vector<bool> ex(n);
        for(int j=0;j<x.size();j++){
            if(ex[x[j].first]) goto end_loop; // inconsistent
            ex[x[j].first]=true;
        }
        
        features_good.push_back(features[i]);
    end_loop:
        continue;
    }

    cout<<">"<<features_good.size()<<" good CCs found"<<endl;
    vector<Point2f> srcs,dsts;
    
    for(int i=0;i<features_good.size();i++){
        vector<pair<int,int>>& fs=features_good[i];
        if(fs.size()!=2 || fs[0].first==fs[1].first){
            cerr<<"unexpected thing happened"<<endl;
            return;
        }
        
        if(fs[0].first==0)
            srcs.push_back(kpss[0][fs[0].second].pt);
        else
            dsts.push_back(kpss[1][fs[0].second].pt);
        
        if(fs[1].first==0)
            srcs.push_back(kpss[0][fs[1].second].pt);
        else
            dsts.push_back(kpss[1][fs[1].second].pt);
    }
    
    
    // visualize feature
    int wmax=0;
    int hsum=0;
    for(int i=0;i<n;i++){
        wmax=(wmax>imgs[i].size().width)?wmax:imgs[i].size().width;
        hsum+=imgs[i].size().height;
    }
    
    Mat vis(hsum,wmax,CV_8UC3);
    int yaccum=0;
    
    vector<int> ydelta;
    for(int i=0;i<n;i++){
        int h=imgs[i].size().height;
        int w=imgs[i].size().width;
        
        Mat dm=vis(Range(yaccum,yaccum+h),Range(0,w));
        imgs[i].copyTo(dm);
        
        ydelta.push_back(yaccum);
        yaccum+=h;
    }
    
    for(int i=0;i<features_good.size();i++){
        vector<pair<int,int>>& fs=features_good[i];
        
        for(int j=0;j<fs.size();j++){
            KeyPoint& kp=kpss[fs[j].first][fs[j].second];
            circle(vis,kp.pt+Point2f(0,ydelta[fs[j].first]),10,0);
        }
        
        line(vis,srcs[i],dsts[i]+Point2f(0,ydelta[1]),0);
    }
    
    imwrite("vis.jpeg",vis);
    

    // use RANSAC (k=3)
    boost::mt19937 rng;
    

    
    const int m=srcs.size();
    const int n_trial=2000;
    const int k_sample=4;
    const int epsilon=3;
    
    double e_best=1e100;
    Mat m_best;
    
    cout<<"RANSAC"<<endl;
    for(int i=0;i<n_trial;i++){
        vector<int> is=random_subset(rng,m,k_sample);
        
        Point2f ss[k_sample],ds[k_sample];
        for(int j=0;j<k_sample;j++){
            ss[j]=srcs[is[j]];
            ds[j]=dsts[is[j]];
        }
        
        Mat m_sd=getPerspectiveTransform(ss,ds);
        
        // eval
        double e=0;
        for(int j=0;j<m;j++){
            Mat ss_ext=Mat(3,1,CV_64F);
            ss_ext.at<double>(0)=srcs[j].x;
            ss_ext.at<double>(1)=srcs[j].y;
            ss_ext.at<double>(2)=1;
            
            Mat ds_cv=Mat(2,1,CV_64F);
            ds_cv.at<double>(0)=dsts[j].x;
            ds_cv.at<double>(1)=dsts[j].y;
            
            Mat s_proj=m_sd*ss_ext;
            Mat s_nhom=Mat(2,1,CV_64F);
            s_nhom.at<double>(0)=s_proj.at<double>(0)/s_proj.at<double>(2);
            s_nhom.at<double>(1)=s_proj.at<double>(1)/s_proj.at<double>(2);
            
            // e+=pow(norm(m_sd*ss_ext,ds_cv),1);
            if(norm(s_nhom,ds_cv)>epsilon) e++;
        }
        
        // update
        if(e<e_best){
            e_best=e;
            m_sd.copyTo(m_best);
        }
        
        cout<<">>iter:"<<i<<"  /  e="<<e<<endl;
    }
    
    cout<<">final error="<<e_best<<endl;
    
    
    Mat fin(imgs[1]);
    
    warpPerspective(imgs[0],fin,m_best,fin.size(),INTER_LINEAR,BORDER_TRANSPARENT);
    
    
    
    imwrite(q.string(),fin);
    
    

    /*
    Mat srcpt(srcs),dstpt(dsts);
    
    cout<<srcpt<<"=----------------------------="<<endl<<dstpt<<endl;
    
//    Mat aff=estimateRigidTransform(srcpt,dstpt,true);
    cout<<"affined transform:"<<aff<<endl;
    
    warpAffine(imgs[0],imgs[1],aff,Size(2000,2000));
    
    imwrite("aff-cmp.jpeg",imgs[1]);
    */
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

void create_index(string dir_path){
    cout<<start_bold<<"creating index"<<end_bold<<endl;
    
    fs::path p(dir_path);
    
    const int min_dim=100; // reject images under this size
    
    int n=0;
    
    ofstream index("./index",ios::out);
    
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
        }
    }
    
    index.close();
    
    cout<<n<<" images are indexed"<<endl;
}

vector<pair<Mat,string>> load_index(){
    cout<<start_bold<<"loading index"<<end_bold<<endl;

    ifstream index("./index",ios::in);
    
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

template<typename K>
bool comparing_values_in(vector<K>& vs,int i,int j){
    return vs[i]<vs[j];
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



// parse argument and pass configuration to the composition function
int main(int argc,char *argv[]){
    vector<pair<Mat,string>> index;
    
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
            create_index("/data/flickr-large");
            index=load_index();
        }
        else if(command=="load"){
            index=load_index();
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

