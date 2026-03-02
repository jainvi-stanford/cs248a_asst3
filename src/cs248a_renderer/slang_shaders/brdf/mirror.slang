implementing "../brdf.slang";

public struct MirrorBRDF
{
    public float3 albedo;

    public float3 evaluate()
    {
        return albedo;
    }

    public float3 sample(float3 rayOut, float3 normal)
    {
        float3 sampleDirection = rayOut;
        float3x3 worldToLocalMatrix = getWorldToLocalMatrix(normal, rayOut);
        float3x3 localToWorldMatrix = transposeMatrix(worldToLocalMatrix);

        // HINT: First convert the reflected ray direction to the local space,
        // then compute the incident ray direction in the local space,
        // and finally convert it to the world space.

        // TODO: Student implementation starts here.
        float3 reflectedDirection = reflect(rayOut, normal);
        float3 localReflectedDirection = mul(worldToLocalMatrix, reflectedDirection);
        float3 localIncidentDirection = -localReflectedDirection;
        sampleDirection = mul(localToWorldMatrix, localIncidentDirection);
        // TODO: Student implementation ends here.

        return sampleDirection;
    }

    public float pdf(float3 rayIn, float3 rayOut, float3 normal)
    {
        const float EPSILON = 1e-4;
        float pdf = EPSILON; // Default to epsilon instead of 0 to avoid fireflies
        float3x3 worldToLocalMatrix = getWorldToLocalMatrix(normal, rayOut);

        // TODO: Student implementation starts here.
        float3 reflectedDirection = reflect(rayOut, normal);
        float3 localReflectedDirection = mul(worldToLocalMatrix, reflectedDirection);
        float3 localIncidentDirection = -localReflectedDirection;
        float3 localRayIn = mul(worldToLocalMatrix, rayIn);
        if (all(abs(localRayIn - localIncidentDirection) < EPSILON))
        {
            pdf = 1.0;
        }
        else
        {            
            pdf = EPSILON;
        }
        // TODO: Student implementation ends here.

        return pdf;
    }
}
