implementing "../primitive.slang";
import math;
import texture;

// The size of each marching step.
static float STEP_SIZE = 0.5;
// The density threshold to consider a hit.
static float DENSITY_THRESHOLD = 0.25;
// The delta used for finite difference normal estimation.
static float FINITE_DIFF_DELTA = 0.001;

public struct Volume
{
    public BoundingBox bound;
    public SharedTexture3D tex;
    public float4x4 modelMatrix;
    public float4x4 invModelMatrix;

    /**
     * Test ray-volume bounding box intersection.
     * @param ray The ray to test against the volume.
     * @param volumeTexBuf The texture buffer contains all the volumetrically represented models in the scene .
     * @return An Optional containing RayHitResult if the ray hits the volume, otherwise an empty Optional.
     */
    public Optional<RayHitResult> hit(Ray ray, SharedTexture3DBuffer<float4> volumeTexBuf)
    {
        // TODO: Student implementation starts here.

        float3 o = mul(invModelMatrix, float4(ray.origin, 1.0)).xyz;
        float3 d = normalize(mul(invModelMatrix, float4(ray.direction, 0.0)).xyz);

        Ray transf_ray = Ray(o, d, ray.tRange);

        // find intersection points within the bounding box
        float3 invDir = 1.0 / transf_ray.direction;
        float3 t0 = (bound.pMin - transf_ray.origin) * invDir;
        float3 t1 = (bound.pMax - transf_ray.origin) * invDir;

        // find which one is closer
        float3 t_close = min(t0, t1);
        float3 t_far = max(t0, t1);

        float t_enter = max(max(t_close.x, t_close.y), t_close.z);
        float t_exit = min(min(t_far.x, t_far.y), t_far.z);

        t_enter = max(t_enter, transf_ray.tRange.x);
        t_exit = min(t_exit, transf_ray.tRange.y);

        if (t_enter > t_exit) return Optional<RayHitResult>();

        float t = t_enter;

        while (t <= t_exit) {
            float3 pt = transf_ray.origin + t * transf_ray.direction;
            float3 uvw = getUVW(pt);

            float4 s = volumeTexBuf.trilinearSample(tex, uvw);
            float density = s.w;
            if (density >= DENSITY_THRESHOLD) {
                    float3 du = float3(FINITE_DIFF_DELTA, 0.0, 0.0);
                    float3 dv = float3(0.0, FINITE_DIFF_DELTA, 0.0);
                    float3 dw = float3(0.0, 0.0, FINITE_DIFF_DELTA);

                    float dx1 = volumeTexBuf.trilinearSample(tex, getUVW(pt - du)).w;
                    float dy0 = volumeTexBuf.trilinearSample(tex, getUVW(pt + dv)).w;
                    float dy1 = volumeTexBuf.trilinearSample(tex, getUVW(pt - dv)).w;
                    float dx0 = volumeTexBuf.trilinearSample(tex, getUVW(pt + du)).w;
                    float dz0 = volumeTexBuf.trilinearSample(tex, getUVW(pt + dw)).w;
                    float dz1 = volumeTexBuf.trilinearSample(tex, getUVW(pt - dw)).w;

                    float3 normal = normalize(float3(dx1 - dx0, dy1 - dy0, dz1 - dz0));
                    float3 normal_world = normalize(mul(float3x3(modelMatrix), normal));

                    return Optional<RayHitResult>(RayHitResult(t, normal_world));                
            }
            t += STEP_SIZE;
        }


        // TODO: Student implementation ends here.

        return Optional<RayHitResult>();
    }

    /**
     * Get the uvw coordinates of a point in local space.
     * @param p The point in world space.
     * @return The uvw coordinates in [0, 1].
     */
    public float3 getUVW(float3 p)
    {
        // Compute uvw coordinates in [0, 1].
        float3 minBound = this.bound.pMin;
        float3 maxBound = this.bound.pMax;
        return (p - minBound) / (maxBound - minBound);
    }
};