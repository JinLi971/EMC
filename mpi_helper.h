#ifndef MPIHELPER_H
#define MPIHELPER_H

#include <mpi.h>
#include "configs.h"
#include <spimage.h>


using namespace MPI;

class MPIHelper
{
public:
    MPIHelper();

    void NewAllTypes(int);
    //pack a model packModel(N_3d, inbuffer, outbuffer)

    void Create_Sp_3Matrix(int);
    void Create_Configuration();
    void Create_Sp_Matrix(int);
    void Create_Sp_Imatrix(int);
    void Create_Complex();
    void Broadcast_Try(int *, int);
    void Broadcast_Masks(sp_imatrix** masks,int N_images, int taskid);
    void Broadcast_Images(sp_matrix ** images, int N_images,int taskid);
    void Broadcast_Mask(sp_imatrix* mask, int taskid);
    void Broadcast_Coordinate(sp_matrix * coor, int taskid);
    void Broadcast_Model(sp_3matrix *model, int taskid);
    void Broadcast_Weight(sp_3matrix * model, int taskid);

    void Broadcast_real(real* vector, int len, int taskid);
    void Broadcast_Config(ConfigD* conf, int taskid);
    void Broadcast_3matrix(sp_3matrix *ma, int taskid);
    void Send_Model(sp_3matrix* model, int root_taskid,int flag);
    void Send_3matrix(sp_3matrix * ma, int taskid, int flag);
    int Recv_3matrix(int len,int root,int flag, sp_3matrix* ma);

    void Send_Respons(real* res, int root, int len);
    void Send_Real(real* res,int master, int len,int a);
    //int Recv_Respons(real* res, int root, int partLen);
    void Recv_Respons(real* res, int root, int partLen, int N_images, int offset);
    void Recv_Respons(int len, int N_images,real* tmp);
    real* Recv_Respons(int partLen, int N_iamges);
    void Recv_Real(int partLen, int N_iamges, int a, real* returnTmp);
    void Send_Respons(real* res,int master, int len, int rank);
    real* Recv_Respons( int* lens, int N_images, int * rank);

    void Global_Allreduce(real* sendbuf, real*outbuf, int count, MPI_Op op );
    //void Gather_Respons


    //Datatype MPI_CONFIG;
    //MPI:: Datatype MPI_COMPLEX;
    //Datatype MPI_SP_3MATRIX;
    //Datatype MPI_SP_IMATRIX;
    //Datatype MPI_SP_MATRIX;
    MPI_Datatype MPI_CONFIG;
    MPI_Datatype MPI_SP_3MATRIX;
    MPI_Datatype MPI_SP_IMATRIX;
    MPI_Datatype MPI_SP_MATRIX;
};
#endif // MPIHELPER_H
