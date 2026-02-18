implementing "../renderer.slang";

import math;
import texture;

public static uint MAX_STEPS = 1024;
public static float EPSILON = 1e-6;

public static float STEP_SIZE = 0.01;
public static float DENSITY_SCALE = 5.0;

/**
 * Perform ray marching through the volume and accumulate color.
 * @param uv The normalized uv coordinate from (0, 0) in the bottom-left to (1, 1) in the top-right.
 * @param uniforms The renderer uniform containing camera, sampling, and scene information.
 * @param tMax The maximum t value for ray marching (usually the nearest primitive intersection t).
 * @param primColor The color of the nearest primitive intersection.
 * @return The accumulated color after ray marching through the volume.
 */
public float4 volumeSample(float2 uv, RendererUniform uniforms, float tMax, float4 primColor)
{
    // TODO: Student implementation starts here.
    Ray ray = uniforms.camera.generateRay(uv);
    Ray localRay = ray.transform(uniforms.volume.invModelMatrix);
    BoundingBox volumeBound = uniforms.volume.bound;
    DiffTexture3D<float, 4> data = uniforms.volume.tex;
    float4 backgroundColor = primColor;
    float4 volumeColor = accumulateColor(localRay, volumeBound, MAX_STEPS, STEP_SIZE, DENSITY_SCALE, tMax, backgroundColor, data);
    return volumeColor;
    
    // TODO: Student implementation ends here.
}

/**
 * Backward pass for volumetric ray marching at a single uv.
 * @param uv The normalized uv coordinate from (0, 0) in the bottom-left to (1, 1) in the top-right.
 * @param uniforms The renderer uniform containing camera, sampling, and scene information.
 * @param outGrad The output gradient w.r.t. the sampled color.
 */
public void volumeSampleBwd(float2 uv, RendererUniform uniforms, float4 outGrad)
{
    // TODO: Student implementation starts here.
    
    // Must use the same tMax and backgroundColor as the forward pass
    float tMax = float.maxValue;  // No mesh intersection in volume recovery
    float4 backgroundColor = uniforms.ambientColor;
    
    Ray ray = uniforms.camera.generateRay(uv);
    Ray localRay = ray.transform(uniforms.volume.invModelMatrix);
    BoundingBox volumeBound = uniforms.volume.bound;
    
    DifferentialPtrPair<DiffTexture3D<float, 4>> diffTexPair = 
        DifferentialPtrPair<DiffTexture3D<float, 4>>(uniforms.volume.tex, uniforms.volume.dTex);
    bwd_diff(accumulateColor)(
        localRay,           // no_diff Ray
        volumeBound,        // no_diff BoundingBox
        MAX_STEPS,          // no_diff uint maxSteps
        STEP_SIZE,          // no_diff float stepSize
        DENSITY_SCALE,      // no_diff float densityScale
        tMax,               // no_diff float tMax
        backgroundColor,    // no_diff float4 backgroundColor
        diffTexPair,        // DifferentialPtrPair<DiffTexture3D>
        outGrad             // output gradient
    );
    // TODO: Student implementation ends here.
}

/**
 * Ray march through the volume and accumulate color.
 * @param ray The ray in local space.
 * @param volumeBound The bounding box of the volume in local space.
 * @param maxSteps The maximum number of marching steps.
 * @param stepSize The size of each marching step.
 * @param densityScale The scale factor for the volume density (alpha channel).
 * @param backgroundColor The background color behind the volume
                          (this can be the ambient color if there's no primitive behind the volume, or the color of the primitive hit if there's one).
 * @param data The 3D texture storing the volume data (rgba).
 * @return The accumulated color after ray marching through the volume.
 */
[Differentiable]
public float4 accumulateColor(
    no_diff Ray ray,
    no_diff BoundingBox volumeBound,
    no_diff uint maxSteps,
    no_diff float stepSize,
    no_diff float densityScale,
    no_diff float tMax,
    no_diff float4 backgroundColor,
    DiffTexture3D<float, 4> data)
{
    // TODO: Student implementation starts here.
    float3 accumColor = float3(0.0);
    float transmittance = 1.0;
    float2 hitRange = volumeBound.hit(ray);
    if (hitRange.y < hitRange.x)
    {
        return backgroundColor;
    }

    float t = max(hitRange.x, ray.tRange.x);
    float tEnd = min(hitRange.y, min(ray.tRange.y, tMax));

    [MaxIters(1024)]
    for (uint i = 0; i < maxSteps && t <= tEnd; i++) {
        float3 samplePoint = ray.origin + t * ray.direction;
        float3 uvw = (samplePoint - volumeBound.pMin) / (volumeBound.pMax - volumeBound.pMin);
        uvw = clamp(uvw, float3(0.0), float3(1.0));
        float4 sampleColor = sampleTrilinear(data, uvw);
        float alpha = 1.0 - exp(-sampleColor.a * densityScale * stepSize);
        float3 emission = sampleColor.rgb;
        accumColor += transmittance * emission * alpha;
        transmittance *= (1.0 - alpha);
        
        // Early termination when transmittance is negligible
        // if (transmittance < 0.001)
        //     break;
            
        t += stepSize;
    }

    float3 finalColor = accumColor + transmittance * backgroundColor.rgb;
    return float4(finalColor, 1.0);

    // TODO: Student implementation ends here.
}
