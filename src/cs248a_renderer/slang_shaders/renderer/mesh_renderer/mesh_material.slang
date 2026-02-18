implementing "../renderer.slang";

import utils;

public float2 getShiftedTexUV(float2 uv, RendererUniform uniforms, Triangle triangle, float2 delta, float2 tex_uv)
{
    float2 shifted_uv = uv + delta;
    Ray ray = uniforms.camera.generateRay(shifted_uv);
    Optional<RayHitResult> hitResult = triangle.hit(ray);
    float2 result = tex_uv;
    if (hitResult.hasValue)
    {
        float3 hitPoint = ray.origin + hitResult.value.t * ray.direction;
        float3 barycentricCoord = triangle.calculateBarycentricCoord(hitPoint);
        return triangle.getUV(barycentricCoord);
    }

    Optional<RayMeshIntersectionResult> rayMeshIntersectionResult_dx = rayMeshIntersection(ray, uniforms);
    if (rayMeshIntersectionResult_dx.hasValue)
    {
        Triangle hitTriangle = rayMeshIntersectionResult_dx.value.hitTriangle;
        if (hitTriangle.getMaterialId() == triangle.getMaterialId())
        {
            float3 hitPoint = ray.origin + rayMeshIntersectionResult_dx.value.closestHit.t * ray.direction;
            float3 barycentricCoord = hitTriangle.calculateBarycentricCoord(hitPoint);
            return hitTriangle.getUV(barycentricCoord);
        }
    }

    return tex_uv;
}

public float getLevel(float2 uv, RendererUniform uniforms, Triangle triangle, float2 tex_uv)
{
    float level = 0.0;

    // TODO: Student implementation starts here.
    float2 dudx = float2(0.0, 0.0);
    float2 dudy = float2(0.0, 0.0);
    float2 pixelStep = float2(1.0 / float(uniforms.camera.canvasSize.x), 1.0 / float(uniforms.camera.canvasSize.y));

    bool hit_dx_valid = false;
    bool hit_dy_valid = false;

    float2 uv_dx = clamp(uv + float2(pixelStep.x, 0.0), float2(0.0, 0.0), float2(1.0, 1.0));
    Ray ray_dx = uniforms.camera.generateRay(uv_dx);
    Optional<RayHitResult> hit_dx = triangle.hit(ray_dx);
    if (hit_dx.hasValue)
    {
        float3 hitPoint_dx = ray_dx.origin + hit_dx.value.t * ray_dx.direction;
        float3 barycentric_dx = triangle.calculateBarycentricCoord(hitPoint_dx);
        float2 tex_uv_dx = triangle.getUV(barycentric_dx);
        dudx = tex_uv_dx - tex_uv;
        hit_dx_valid = true;
    }

    float2 uv_dy = clamp(uv + float2(0.0, pixelStep.y), float2(0.0, 0.0), float2(1.0, 1.0));
    Ray ray_dy = uniforms.camera.generateRay(uv_dy);
    Optional<RayHitResult> hit_dy = triangle.hit(ray_dy);
    if (hit_dy.hasValue)
    {
        float3 hitPoint_dy = ray_dy.origin + hit_dy.value.t * ray_dy.direction;
        float3 barycentric_dy = triangle.calculateBarycentricCoord(hitPoint_dy);
        float2 tex_uv_dy = triangle.getUV(barycentric_dy);
        dudy = tex_uv_dy - tex_uv;
        hit_dy_valid = true;
    }
    
    // If one derivative is missing, use the other one for both
    if (!hit_dx_valid && hit_dy_valid) {
        dudx = dudy;
    } else if (!hit_dy_valid && hit_dx_valid) {
        dudy = dudx;
    }
    
    float maxDerivative = max(length(dudx), length(dudy));
    uint materialId = triangle.materialId + triangle.offset;
    PhysicsBasedMaterial material = uniforms.physicsBasedMaterialBuf[materialId];
    uint2 texSize = material.albedo.mipmap[0].size;
    float maxMipLevel = float(material.albedo.totalLevels - 1);
    level = log2(maxDerivative * float(max(texSize.x, texSize.y)));
    level = clamp(level, 0.0, maxMipLevel);

    return level;
    // TODO: Student implementation ends here.

    return level;
}

public PhysicsBasedParameters getHitPointParameters(float2 uv, RendererUniform uniforms, Triangle triangle, float3 normal, float3 barycentricCoord, float3 hitPoint)
{
    float2 tex_uv = triangle.getUV(barycentricCoord);
    float level = getLevel(uv, uniforms, triangle, tex_uv);
    uint materialId = triangle.materialId + triangle.offset;
    PhysicsBasedMaterial physicsBasedMaterial = uniforms.physicsBasedMaterialBuf[materialId];
    PhysicsBasedParameters parameters = sampleMaterial(physicsBasedMaterial, uniforms.physicsBasedMaterialTextureBuf, tex_uv, level, uniforms.visualizeLevelOfDetail);
    if (uniforms.visualizeBarycentricCoords)
    {
        parameters.albedo = barycentricCoord;
    }
    if (uniforms.visualizeTexUV)
    {
        parameters.albedo = float3(tex_uv, 0.0);
    }
    return parameters;
}
