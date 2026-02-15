implementing "../brdf.slang";

public struct GlassBRDF
{
    public float3 albedo;
    public float ior;

    public float3 evaluate()
    {
        return albedo;
    }

    public float3 sample(float3 rayOut, float3 normal)
    {

        float3 sampleDirection = rayOut;
        float3x3 worldToLocalMatrix = getWorldToLocalMatrix(normal, rayOut);
        float3x3 localToWorldMatrix = transposeMatrix(worldToLocalMatrix);
        
        // TODO: Student implementation starts here.

        // TODO: Student implementation ends here.

        return sampleDirection;
    }

    public float pdf(float3 rayIn, float3 rayOut, float3 normal)
    {
        // For dielectric materials, the PDF is 1.0 for the single direction
        // (either refracted or reflected due to total internal reflection)
        return 1.0;
    }
}
