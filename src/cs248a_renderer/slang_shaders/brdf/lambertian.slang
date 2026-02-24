implementing "../brdf.slang";

import utils;

public struct LambertianBRDF
{
    public float3 albedo;
    public bool useCosineWeightedSampling;

    public float3 evaluate()
    {
        float3 fr = this.albedo;

        fr = this.albedo * INV_PI;

        return fr;
    }

    public float3 sample(float3 rayOut, float3 normal, float2 randomEpsilon)
    {
        float3 sampleDirection = rayOut;
        float3x3 worldToLocalMatrix = getWorldToLocalMatrix(normal, rayOut);
        float3x3 localToWorldMatrix = transposeMatrix(worldToLocalMatrix);

        // HINT: First convert the reflected ray direction to the local space,
        // then compute the incident ray direction in the local space,
        // and finally convert it to the world space.
        float3 localSampleDirection = mul(worldToLocalMatrix, sampleDirection);
        

        if (this.useCosineWeightedSampling)
        {
            // TODO: Student implementation starts here.
            float phi = 2.0 * PI * randomEpsilon.x;
            float cosTheta = sqrt(randomEpsilon.y); // Inverse CDF for cosine-weighted hemisphere sampling
            float sinTheta = sqrt(1.0 - cosTheta * cosTheta);
            // Convert spherical to Cartesian (local space, y-up since normal is y-axis)
            localSampleDirection = float3(
                sinTheta * cos(phi),   // x
                cosTheta,              // y is "up" in local space (along normal)
                sinTheta * sin(phi)    // z
            );
            sampleDirection = mul(localToWorldMatrix, localSampleDirection);

            // TODO: Student implementation ends here.
        }
        else // uniform sampling, use uniform sampling by default
        {
            // TODO: Student implementation starts here.
            float phi = 2.0 * PI * randomEpsilon.x;
            float cosTheta = randomEpsilon.y;
            float sinTheta = sqrt(1.0 - cosTheta * cosTheta);
            
            // Convert spherical to Cartesian (local space, y-up since normal is y-axis)
            localSampleDirection = float3(
                sinTheta * cos(phi),   // x
                cosTheta,              // y is "up" in local space (along normal)
                sinTheta * sin(phi)    // z
            );
            sampleDirection = mul(localToWorldMatrix, localSampleDirection);

            // TODO: Student implementation ends here.
        }

        return sampleDirection;
    }

    public float pdf(float3 rayIn, float3 rayOut, float3 normal)
    {
        float EPSILON = 1e-6;
        float pdf = all(abs(rayIn - rayOut) < EPSILON) ? 1.0 : 0.0;

        if (this.useCosineWeightedSampling)
        {
            // TODO: Student implementation starts here.
            // rayIn is the sampled direction, so we compute cosTheta with it
            float cosTheta = max(dot(rayIn, normal), 0.0);
            pdf = cosTheta * INV_PI;

            // TODO: Student implementation ends here.
        }
        else // uniform sampling, use uniform sampling by default
        {
            // TODO: Student implementation starts here.
            // Uniform hemisphere sampling has constant PDF = 1/(2*PI)
            pdf = 1.0 / (2.0 * PI);

            // TODO: Student implementation ends here.
        }

        return pdf;
    }
}
