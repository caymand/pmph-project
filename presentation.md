-- A[M][K], B[K][N], C[M][N]
for blockM = 0 to M step warps_m * wmma_m do -- par
  for blockN = 0 to N step warps_n * wmma_n do -- par
    for warpM = 0 to warpsM do -- par
      for warpN = 0 to warpsN do  -- par
	    fragment c = 0 -- (wmma_m, wmma_n)
	    for blockK = 0 to K step wmma_k do -- seq
	      fragment a = load_matrix_sync(A[blockM + warp_m * wmma_m][blockK ]
		  fragment b = load_matrix_sync(B[blockK ][blockN + warp_n * wmma_n]
		  mma(c, a, b, c)
        store_matrix(C[...][...], c)
			

