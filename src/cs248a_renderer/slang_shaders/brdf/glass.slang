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
        // Using notebook formula: w_i = -eta * w_o + (eta * cosTheta1 - cosTheta2) * n
        // where cosTheta1 = w_o · n
        float cosTheta1 = dot(rayOut, normal);
        float3 n = normal;
        float eta;
        
        if (cosTheta1 > 0.0)
        {
            // Entering glass (air to glass): eta = n1/n2 = 1/ior
            eta = 1.0 / ior;
        }
        else
        {
            // Exiting glass (glass to air): eta = n1/n2 = ior
            eta = ior;
            cosTheta1 = -cosTheta1;
            n = -normal;
        }
        
        float sinTheta1Sq = 1.0 - cosTheta1 * cosTheta1;
        float sinTheta2Sq = eta * eta * sinTheta1Sq;
        
        if (sinTheta2Sq > 1.0)
        {
            // Total internal reflection
            sampleDirection = reflect(rayOut, n);
        }
        else
        {
            // Refraction using Snell's law
            float cosTheta2 = sqrt(1.0 - sinTheta2Sq);
            sampleDirection = -eta * rayOut + (eta * cosTheta1 - cosTheta2) * n;
        }
        sampleDirection = normalize(sampleDirection);
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
