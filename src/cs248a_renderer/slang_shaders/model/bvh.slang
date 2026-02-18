implementing "../model.slang";
import math;

public struct BVHNode
{
    public BoundingBox bound;
    public int left;
    public int right;
    public int primLeft;
    public int primRight;
    public int depth;

    public __init(BoundingBox bound, int left = -1, int right = -1, int primLeft = -1, int primRight = -1, int depth = -1)
    {
        this.bound = bound;
        this.left = left;
        this.right = right;
        this.primLeft = primLeft;
        this.primRight = primRight;
        this.depth = depth;
    }

    public float2 hit(Ray ray)
    {
        return this.bound.hit(ray);
    }
}

public struct BVH<T>
    where T : IRayCastable
{
    public struct HitResult
    {
        public T primitive;
        public RayHitResult rayHitResult;
    }

    // Slang buffer of BVHNode copied from Python.
    public RWStructuredBuffer<BVHNode> nodes;
    // Maximum number of nodes in the BVH, in this case, it'll be the size of the nodes buffer.
    public uint maxNodes;
    // Slang buffer of primitives represented by the BVH.
    public RWStructuredBuffer<T> primitives;
    // Number of primitives in the primitives buffer.
    public uint numPrimitives;

    public __init(RWStructuredBuffer<BVHNode> nodes, RWStructuredBuffer<T> primitives, uint numPrimitives, uint maxNodes)
    {
        this.nodes = nodes;
        this.primitives = primitives;
        this.numPrimitives = numPrimitives;
        this.maxNodes = maxNodes;
    }

    struct HitStack<int N>
    {
        uint[N] nodeIndices;
        uint top;

        __init()
        {
            this.top = 0;
        }

        [mutating]
        void push(uint nodeIndex)
        {
            this.nodeIndices[this.top] = nodeIndex;
            this.top += 1;
        }

        [mutating]
        uint pop()
        {
            this.top -= 1;
            return this.nodeIndices[this.top];
        }

        bool isEmpty()
        {
            return this.top == 0;
        }

        uint size()
        {
            return this.top;
        }
    }

    // BVH traversal to find the closest hit.
    public Optional<HitResult> hit(Ray ray)
    {
        // Initialize traversal.
        Optional<HitResult> hitResult = Optional<HitResult>();
        HitStack<64> stack;

        // TODO: Student implementation starts here.
        stack.push(0); // Start with root node
        float closestT = ray.tRange.y;
        while (!stack.isEmpty())
        {
            uint nodeIndex = stack.pop();
            BVHNode node = this.nodes[nodeIndex];

            float2 tHit = node.hit(ray);
            if (tHit.y >= tHit.x)
            {
                // Leaf node
                if (node.left == -1 && node.right == -1)
                {
                    for (int i = node.primLeft; i < node.primRight; i++)
                    {
                        T primitive = this.primitives[i];
                        Optional<RayHitResult> localHit = primitive.hit(ray);
                        if (localHit.hasValue && localHit.value.t < closestT)
                        {
                            closestT = localHit.value.t;
                            ray.tRange.y = closestT;
                            HitResult result;
                            result.primitive = primitive;
                            result.rayHitResult = localHit.value;
                            hitResult = Optional<HitResult>(result);
                        }
                    }
                }
                else
                {
                    // Internal node, push children onto stack
                    if (node.left != -1)
                    {
                        stack.push(node.left);
                    }
                    if (node.right != -1)
                    {
                        stack.push(node.right);
                    }
                }
            }
        }
        // TODO: Student implementation ends here.

        return hitResult;
    }
}
