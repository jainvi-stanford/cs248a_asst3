implementing "../primitive.slang";
import math;

public struct Triangle : IBoundingBox, IRayCastable
{
    public float3[3] vertices;
    public float3[3] normals;
    public float2[3] uvs;
    public float3[3] colors;
    public uint materialId;
    public uint offset;

    public BoundingBox boundingBox()
    {
        return BoundingBox(float3(0.0), float3(0.0));
    }

    /* Returns the average color of the triangle. */
    public float3 averageColor()
    {
        return (this.colors[0] + this.colors[1] + this.colors[2]) / 3.0;
    }

    /**
     * Calculates the barycentric coordinates of a point on the triangle.
     * @param p The point on the triangle to calculate the barycentric coordinates for.
     * @return The barycentric coordinates of the point.
     */
    public float3 calculateBarycentricCoord(float3 p)
    {
        const float EPSILON = 1e-6;
        float u = 0.0;
        float v = 0.0;
        float w = 0.0;

        // TODO: Student implementation starts here.
        float3 v0 = vertices[1] - vertices[0];
        float3 v1 = vertices[2] - vertices[0];
        float3 v2 = p - vertices[0];
        float d00 = dot(v0, v0);
        float d01 = dot(v0, v1);
        float d11 = dot(v1, v1);
        float d20 = dot(v2, v0);
        float d21 = dot(v2, v1);
        float denom = d00 * d11 - d01 * d01;
        if (abs(denom) < EPSILON) {
            return float3(-1.0, -1.0, -1.0); // Degenerate triangle
        }
        v = (d11 * d20 - d01 * d21) / denom;
        w = (d00 * d21 - d01 * d20) / denom;
        u = 1.0 - v - w;
        // TODO: Student implementation ends here.

        return float3(u, v, w);
    }

    public float2 getUV(float3 barycentricCoord)
    {
        float2 uv = float2(0.0);

        float u = barycentricCoord.x;
        float v = barycentricCoord.y;
        float w = barycentricCoord.z;
        uv = u * this.uvs[0] + v * this.uvs[1] + w * this.uvs[2];
        
        return uv;
    }

    public float3 getColor(float3 barycentricCoord)
    {
        float u = barycentricCoord.x;
        float v = barycentricCoord.y;
        float w = barycentricCoord.z;
        float3 color = u * this.colors[0] + v * this.colors[1] + w * this.colors[2];
        return color;
    }

    public float3 getNormal(float3 barycentricCoord)
    {
        float u = barycentricCoord.x;
        float v = barycentricCoord.y;
        float w = barycentricCoord.z;
        float3 normal = u * this.normals[0] + v * this.normals[1] + w * this.normals[2];
        return normalize(normal);
    }

    /**
     * Test ray-triangle intersection.
     * @param ray The ray to test against the triangle.
     * @return An Optional containing RayHitResult if the ray hits the triangle, otherwise an empty Optional.
     */
    public Optional<RayHitResult> hit(Ray ray)
    {
        // TODO: Student implementation starts here.
        const float EPSILON = 1e-6;

        float3 e1 = vertices[1] - vertices[0];
        float3 e2 = vertices[2] - vertices[0];

        float3 normal = normalize(cross(e1, e2));
        float3 d_norm = normalize(ray.direction);

        // Use normalized vectors for the parallel check
        float det = dot(normal, d_norm);
        if (abs(det) < EPSILON){
            return Optional<RayHitResult>();
        }

        // Compute t using the original ray direction so caller can use it directly
        float3 p0 = ray.origin - vertices[0];
        float denom_t = dot(normal, ray.direction);
        float t = -dot(normal, p0) / denom_t;
        
        // Compute hit point and barycentric coordinates
        // Use same convention as calculateBarycentricCoord: e1 first, then e2
        float3 intP = ray.origin + t * ray.direction;
        float3 vp = intP - vertices[0];

        float d00 = dot(e1, e1);
        float d01 = dot(e1, e2);
        float d11 = dot(e2, e2);
        float d20 = dot(vp, e1);
        float d21 = dot(vp, e2);

        float denom = (d00*d11) - (d01*d01);
        if (abs(denom) < EPSILON) {
            return Optional<RayHitResult>(); // Degenerate triangle
        }
        float v = ((d11*d20) - (d01*d21)) / denom;
        float w = ((d00*d21) - (d01*d20)) / denom;
        float u = 1.0 - v - w;

        if ((u >= -EPSILON && v >= -EPSILON) && (w >= -EPSILON)) {
            // Check if t is within the valid ray range [tMin, tMax]
            if (t >= ray.tRange.x && t <= ray.tRange.y) {
                return Optional<RayHitResult>(RayHitResult(t, normal));
            }
        }

        return Optional<RayHitResult>();

        // TODO: Student implementation ends here.
    }

    public uint getMaterialId()
    {
        return this.materialId + this.offset;
    }
}
