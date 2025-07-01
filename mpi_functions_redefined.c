/* This code is needed since some MPI calls in the code 
   need to be adapted to handle cases in which the size 
   of data to read/communicate per task exceeds 2^31 - 1 */

#include "ricklib.h"

/* INT_MAX is the maximum signed integer, defined in limitations.h, included
   in the RICK header file */ 

int MPI_File_read_at_custom(MPI_File fh, MPI_Offset offset, void *buf,
		     myull count, MPI_Datatype datatype, MPI_Status *status)
{

  int ret;
    
  /* Check if count exceeds INT_MAX */
  if (count > INT_MAX)
    {
      MPI_Datatype chunks; /* Split the entire buffer size in smaller chunks */
      MPI_Type_contiguous(INT_MAX, datatype, &chunks); /* All the chunks have dimension INT_MAX */
      MPI_Type_commit(&chunks);

      myull chunk_count = count / INT_MAX;
      myull reminder    = count % INT_MAX; /* Consider the reminder in offset and buffer pointer in reminder > 0 */

      if (reminder > 0)
	ret = PMPI_File_read_at(fh, offset, buf, reminder, datatype, status);

      /* If reminder == 0 this modification does not impact on the result */
      ret = PMPI_File_read_at(fh, offset+reminder, buf+reminder, chunk_count, chunks, status);

      MPI_Type_free(&chunks); /* Free the created datatype */
    }
  else /* Standard case in which count is <= INT_MAX */
    ret = PMPI_File_read_at(fh, offset, buf, count, datatype, status);
  
  return ret;
}

int MPI_Sendrecv_custom(void *sendbuf, myull sendcount, MPI_Datatype sendtype,
			int dest, int sendtag, void *recvbuf, myull recvcount,
			MPI_Datatype recvtype, int source, int recvtag,
			MPI_Comm comm, MPI_Status *status)
{

  int ret;

  /* Define type size for both sendtype and recvtype */
  int send_size, recv_size;

  MPI_Type_size(sendtype, &send_size);
  MPI_Type_size(recvtype, &recv_size);

  MPI_Aint send_bytes = (MPI_Aint)sendcount * send_size;
  MPI_Aint recv_bytes = (MPI_Aint)recvcount * recv_size;
  
  MPI_Datatype new_sendtype, new_recvtype;
  int send_as_one = 0, recv_as_one = 0;
  
  if (send_bytes > INT_MAX) {
    MPI_Type_contiguous(sendcount, sendtype, &new_sendtype);
    MPI_Type_commit(&new_sendtype);
    send_as_one = 1;
  }

  if (recv_bytes > INT_MAX) {
        MPI_Type_contiguous(recvcount, recvtype, &new_recvtype);
        MPI_Type_commit(&new_recvtype);
        recv_as_one = 1;
    }

  ret = PMPI_Sendrecv(send_as_one ? sendbuf : sendbuf, 
	       send_as_one ? 1 : sendcount, 
	       send_as_one ? new_sendtype : sendtype, 
	       dest, sendtag,
	       recv_as_one ? recvbuf : recvbuf, 
	       recv_as_one ? 1 : recvcount, 
	       recv_as_one ? new_recvtype : recvtype, 
	       source, recvtag,
	       comm, status);

  if (send_as_one)
    MPI_Type_free(&new_sendtype);
  

  if (recv_as_one)
    MPI_Type_free(&new_recvtype);
  
  return ret;  
}
