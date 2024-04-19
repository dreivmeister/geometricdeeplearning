function convolution(input, filter)
    # input  of shape (iheight,iwidth,ninpch)
    # filter of shape (noutch,fheight,fwidth,ninpch)
    # output of shape (oheight,owidth,noutch)
    # oheight = iheight - (fheight-1)
    # owidth  = iwidth  - (fwidth-1)

    iheight, iwidth, ninpch = size(input)
    noutch, fheight, fwidth, ninpch = size(filter)
    fh2 = Int((fheight-1)/2)
    fw2 = Int((fwidth-1)/2)
    oheight = iheight-fheight+1
    owdith = iwidth-fwidth+1
    output = Array{Float64}(undef, (oheight,owdith,noutch))

    for u in 1+fh2:fh2+oheight
        for v in 1+fw2:fw2+owdith
            #TODO: could you vectorize over out channels? prolly need to "clone" input j times
            for j in 1:noutch
                # accumulate output at position u,v,j
                output[u-fh2,v-fw2,j] = sum(filter[j,:,:,:] .* input[u-fh2:u+fh2,v-fw2:v+fw2,:])

            end
        end
    end

    return output
end



inp = [0 1 1 1 0 0 0;
        0 0 1 1 1 0 0;
        0 0 0 1 1 1 0;
        0 0 0 1 1 0 0;
        0 0 1 1 0 0 0;
        0 1 1 0 0 0 0;
        1 1 0 0 0 0 0;]

fil = [1 0 1;
        0 1 0;
        1 0 1;]

convolution(reshape(inp, (7,7,1)), reshape(fil, (1,3,3,1)))