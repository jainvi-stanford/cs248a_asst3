implementing "../texture.slang";

public struct SharedTexture2DBuffer<T>
    where T : IFloat
{
    public StructuredBuffer<T> buffer;

    T getBufferValue(SharedTexture2D tex, uint2 texelCoord)
    {
        uint2 texSize = tex.size;
        uint offset = tex.offset;
        uint index = texelCoord.y * texSize.x + texelCoord.x;
        return buffer[offset + index];
    }

    public T pointSample(SharedTexture2D tex, float2 uv)
    {
        T outV = T(0.0);
        // TODO: Student implementation starts here.
        float2 texCoords = uv * float2(tex.size);
        uint2 texelCoord = uint2(clamp(texCoords, float2(0.0), float2(tex.size) - float2(1.0)));
        outV = getBufferValue(tex, texelCoord);
        // TODO: Student implementation ends here.
        return outV;
    }

    public T bilinearSample(SharedTexture2D tex, float2 uv)
    {
        T outV = T(0.0);
        // TODO: Student implementation starts here.
        float2 texCoords = uv * (tex.size - 1);
        uint2 base = uint2(floor(texCoords));
        float2 offset = texCoords - base;
        uint2 i0 = base;
        uint2 i1 = min(i0 + uint2(1, 1), tex.size - uint2(1, 1));
        T c00 = getBufferValue(tex, i0);
        T c10 = getBufferValue(tex, uint2(i1.x, i0.y));
        T c01 = getBufferValue(tex, uint2(i0.x, i1.y));
        T c11 = getBufferValue(tex, i1);
        T x0 = c00 + (c10 - c00) * T(offset.x);
        T x1 = c01 + (c11 - c01) * T(offset.x);
        outV = x0 + (x1 - x0) * T(offset.y);
        // TODO: Student implementation ends here.
        return outV;
    }

    public T trilinearSample(SharedTexture2D[] mipmapLevels, float2 uv, float d, uint totalLevels)
    {
        T outV = T(0.0);
        // TODO: Student implementation starts here.
        float level = clamp(d, 0.0, float(totalLevels - 1));
        uint level0 = uint(floor(level));
        uint level1 = min(level0 + 1, totalLevels - 1);
        float levelOffset = level - float(level0);
        T sample0 = bilinearSample(mipmapLevels[level0], uv);
        T sample1 = bilinearSample(mipmapLevels[level1], uv);
        outV = sample0 + (sample1 - sample0) * T(levelOffset);
        // TODO: Student implementation ends here.
        return outV;
    }
}

public struct SharedTexture2D
{
    public uint2 size;
    public uint offset;
}

public struct SharedTexture3DBuffer<T>
    where T : IFloat
{
    public StructuredBuffer<T> buffer;

    T getBufferValue(SharedTexture3D tex, uint3 voxelCoord)
    {
        uint3 texSize = tex.size;
        uint offset = tex.offset;
        uint index = voxelCoord.z * texSize.y * texSize.x +
                     voxelCoord.y * texSize.x +
                     voxelCoord.x;
        return buffer[offset + index];
    }

    /**
     * Sample the texture at the given uvw coordinates using nearest neighbor interpolation.
     * @param tex The texture to sample.
     * @param uvw The uvw coordinates to sample the texture at.
     * @return The sampled value.
     */
    public T pointSample(SharedTexture3D tex, float3 uvw)
    {
        // TODO: Student implementation starts here.

        T voxel_val = getBufferValue(tex, uint3(uvw * tex.size));

        return voxel_val;

        // TODO: Student implementation ends here.
    }

    /**
     * Sample the texture at the given uvw coordinates using trilinear interpolation.
     * @param tex The texture to sample.
     * @param uvw The uvw coordinates to sample the texture at.
     * @return The sampled value.
     **/
    public T trilinearSample(SharedTexture3D tex, float3 uvw)
    {
        // TODO: Student implementation starts here.
        T voxelVal = getBufferValue(tex, uint3(uvw * tex.size));
        float3 coords = uvw * (tex.size - 1);
        float3 base = floor(coords);
        float3 offset = coords - base;

        int3 i0 = int3(base); // converting base voxel to int
        int3 i1 = min(i0 + int3(1, 1, 1), tex.size - int3(1, 1, 1));

        // getting the 8 coords

        T c000 = getBufferValue(tex, uint3(i0.x, i0.y, i0.z));
        T c100 = getBufferValue(tex, uint3(i1.x, i0.y, i0.z));
        T c010 = getBufferValue(tex, uint3(i0.x, i1.y, i0.z));
        T c110 = getBufferValue(tex, uint3(i1.x, i1.y, i0.z));
        T c001 = getBufferValue(tex, uint3(i0.x, i0.y, i1.z));
        T c101 = getBufferValue(tex, uint3(i1.x, i0.y, i1.z));
        T c011 = getBufferValue(tex, uint3(i0.x, i1.y, i1.z));
        T c111 = getBufferValue(tex, uint3(i1.x, i1.y, i1.z));

        T x00 = c000 + (c100 - c000) * T(offset.x);
        T x10 = c010 + (c110 - c010) * T(offset.x);
        T x01 = c001 + (c101 - c001) * T(offset.x);
        T x11 = c011 + (c111 - c011) * T(offset.x);

        T y0 = x00 + (x10 - x00) * T(offset.y);
        T y1 = x01 + (x11 - x01) * T(offset.y);

        T z = y0 + (y1 - y0) * T(offset.z);

        return z;

        // TODO: Student implementation ends here.
    }
}

public struct SharedTexture3D
{
    public uint3 size;
    public uint offset;
}
