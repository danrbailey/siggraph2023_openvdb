
#include <openvdb/openvdb.h>
#include <openvdb/util/CpuTimer.h>
#include <openvdb/tools/Prune.h>

#include "../asset.h"
#include "../parse.h"

#include <tbb/enumerable_thread_specific.h>

using namespace openvdb;

// These benchmarks were featured in the Siggraph 2023 OpenVDB Course under the talk
// Multi-Threading in OpenVDB:
// https://s2023.siggraph.org/presentation/?id=gensub_180&sess=sess161
// The slides are publicly available under the OpenVDB website and should be referred to
// as they step through each of the techniques demonstrated here.

// Default threshold of 0.5f
const float threshold = 0.5f;

// collapseLeaf() oracle which is functionality equivalent to tools::prune() for leaf nodes
bool collapseLeaf(const FloatTree::LeafNodeType& leaf)
{
    // only dense leafs (all active values) are considered for pruning
    if (!leaf.isDense())    return false;

    // any leafs where the value is less than the threshold should not be pruned
    for (auto iter = leaf.cbeginValueOn(); iter; ++iter) {
        if (*iter < threshold)      return false;
    }
    return true;
}

///////////////////////////////////////////////////////////////////////
// Technique 1: Copy, Clamp and Prune

void copyClampAndPrune(const FloatTree& inputTree, bool threaded = true)
{
    util::CpuTimer timer;
    timer.start("Copy, Clamp and Prune");

    // deep-copy the input const grid so that it can be modified

    FloatTree tree(inputTree);

    if (threaded) {
        // this places the unthreaded algorithm below in a lambda and evaluates it in parallel
        // using a node manager to iterate over the nodes of the tree in breadth-first order
        auto clampOp = [&](auto& node)
        {
            for (auto iter = node.beginValueOn(); iter; ++iter) {
                if (*iter > threshold)    iter.setValue(threshold);
            }
        };
        tree::NodeManager<FloatTree> nodeManager(tree);
        nodeManager.foreachTopDown(clampOp);
    } else {
        // this uses the tree value iterator which is unthreaded and slow
        for (auto iter = tree.beginValueOn(); iter; ++iter) {
            if (*iter > threshold)    iter.setValue(threshold);
        }
    }

    // prune any leaf nodes and internal nodes
    tools::prune(tree);

    timer.stop();
}

///////////////////////////////////////////////////////////////////////
// Technique 2: Mask Build

struct DeepCopyFloatOp
{
    DeepCopyFloatOp(const FloatTree& inputTree)
        : mInputTree(inputTree) { }

    void operator()(FloatTree::RootNodeType& root) const
    {
        // iterate over all source root node tiles and add new tiles with the same
        // value and active state to the target root node
        for (auto iter = mInputTree.root().cbeginValueAll(); iter; ++iter) {
            float value = (*iter > threshold) ? threshold : *iter;
            root.addTile(iter.getCoord(), value, iter.isValueOn());
        }
    }

    template <typename T>
    void operator()(T& node) const
    {
        // find the corresponding node in the source tree and add new tiles
        // with the same value and active state to the target root node
        const auto* sourceNode = mInputTree.probeConstNode<T>(node.origin());
        for (auto iter = sourceNode->cbeginValueAll(); iter; ++iter) {
            float value = (*iter > threshold) ? threshold : *iter;
            node.addTile(iter.pos(), value, iter.isValueOn());
        }
    }

    void operator()(FloatTree::LeafNodeType& leaf) const
    {
        // find the corresponding leaf node in the source tree and update the values
        // of the target leaf node for all active voxels to match
        const auto* sourceLeaf = mInputTree.probeLeaf(leaf.origin());
        for (auto iter = leaf.beginValueOn(); iter; ++iter) {
            float value = sourceLeaf->getValue(iter.pos());
            if (value > threshold)  value = threshold;
            iter.setValue(value);
        }
    }

    const FloatTree& mInputTree;
};

void maskBuild(const FloatTree& inputTree)
{
    util::CpuTimer timer;
    timer.start("Mask Build");

    // deep-copy the topology of the input tree only - this is very fast and lightweight
    MaskTree maskTree(inputTree);

    // iterate over the mask tree in bottom-up, breadth-first order and turn
    // any leaf nodes into tiles that match the conditional logic in collapseLeaf()
    auto deleteMaskLeafsOp = [&](auto& node)
    {
        if constexpr(node.LEVEL == 0) { // leaf node
            const auto* sourceLeaf = inputTree.probeLeaf(node.origin());
            if (collapseLeaf(*sourceLeaf)) {
                // make all voxels in this leaf node inactive
                node.setValuesOff();
            }
        } else if constexpr(node.LEVEL == 1) { // internal node parent
            for (auto iter = node.beginChildOn(); iter; ++iter) {
                // replace with tiles any leaf nodes that have only inactive voxels
                // (as previously set in the phase above ^)
                if (iter->isEmpty())    node.addTile(iter.pos(), true, true);
            }
        }
    };
    tree::NodeManager<MaskTree> maskNodeManager(maskTree);
    maskNodeManager.foreachBottomUp(deleteMaskLeafsOp);

    // create a new float tree to match the new reduced mask topology
    FloatTree tree(maskTree, threshold, TopologyCopy());
    tree.root().setBackground(0.0f, /*updateChildNodes=*/false);

    // deep-copy all the active values from the source tree to the target tree
    tree::NodeManager<FloatTree> nodeManager(tree);
    DeepCopyFloatOp deepCopyFloatOp(inputTree);
    nodeManager.foreachTopDown(deepCopyFloatOp);

    // prune any remaining internal nodes
    tools::pruneTiles(tree);

    timer.stop();
}

///////////////////////////////////////////////////////////////////////
// Technique 3: Grid Reduce

template <bool LeafIteration>
struct ScatterVDBOp
{
    explicit ScatterVDBOp(tbb::enumerable_thread_specific<FloatTree>& pool)
        : mPool(pool) { }

    ScatterVDBOp(const ScatterVDBOp& other, tbb::split)
        : mPool(other.mPool) { }

    bool operator()(const FloatTree::RootNodeType& root, size_t) const
    {
        // set the root tile values of the tree local to this thread to match the source tree
        FloatTree& tree = mPool.local();
        tree::ValueAccessor<FloatTree> acc(tree);
        for (auto iter = root.cbeginValueOn(); iter; ++iter) {
            float value = *iter;
            if (value > threshold)  value = threshold;
            acc.addTile(root.getLevel(), iter.getCoord(), value, true);
        }
        return true;
    }

    template <typename T>
    bool operator()(const T& node, size_t) const
    {
        // set the internal node values of the tree local to this thread to match the source tree
        FloatTree& tree = mPool.local();
        tree::ValueAccessor<FloatTree> acc(tree);
        for (auto iter = node.cbeginValueOn(); iter; ++iter) {
            float value = *iter;
            if (value > threshold)  value = threshold;
            acc.addTile(node.getLevel(), iter.getCoord(), value, true);
        }

        // if direct LeafIteration is disabled, iterate over the leaf nodes during this internal node phase
        if constexpr (node.LEVEL == 1 && !LeafIteration) {
            for (auto leaf = node.cbeginChildOn(); leaf; ++leaf) {
                if (collapseLeaf(*leaf)) {
                    if (leaf->isDense()) {
                        acc.addTile(1, leaf->origin(), threshold, true);
                    }
                } else {
                    for (auto iter = leaf->cbeginValueOn(); iter; ++iter) {
                        float value = (*iter > threshold) ? threshold : *iter;
                        acc.setValue(iter.getCoord(), value);
                    }
                }
            }
        }

        return true;
    }

    bool operator()(const FloatTree::LeafNodeType& leaf, size_t) const
    {
        // if direct LeafIteration is enabled, iterate over the leaf nodes here
        if (LeafIteration) {
            FloatTree& tree = mPool.local();
            tree::ValueAccessor<FloatTree> acc(tree);

            // find any leaf nodes that should be turned into tiles

            if (collapseLeaf(leaf)) {
                if (leaf.isDense()) {
                    acc.addTile(1, leaf.origin(), threshold, true);
                }
            } else {
                for (auto iter = leaf.cbeginValueOn(); iter; ++iter) {
                    float value = (*iter > threshold) ? threshold : *iter;
                    acc.setValue(iter.getCoord(), value);
                }
            }
        }

        return true;
    }

    tbb::enumerable_thread_specific<FloatTree>& mPool;
}; // struct ScatterVDBOp

void gridReduce(const FloatTree& inputTree)
{
    util::CpuTimer timer;
    timer.start("Grid Reduce");

    // create a thread-local storage container using TBB

    FloatTree tree(0.0f);
    tbb::enumerable_thread_specific<FloatTree> pool(tree);

    // scatter into each thread-local tree

    ScatterVDBOp<false> scatterOp(pool);
    tree::DynamicNodeManager<const FloatTree> nodeManager(inputTree);
    nodeManager.foreachTopDown(scatterOp);

    // merge all the thread-local trees into one tree

    for (auto it = pool.begin(); it < pool.end(); ++it) {
        tree.merge(*it);
    }

    // prune any remaining internal nodes

    tools::pruneTiles(tree);

    timer.stop();
}

///////////////////////////////////////////////////////////////////////
// Technique 4: Dynamic Build

struct DynamicBuildOp
{
    explicit DynamicBuildOp(const FloatTree& inputTree)
        : mInputTree(inputTree) { }

    bool operator()(FloatTree::RootNodeType& root, size_t) const
    {
        // iterate over the source tree root child nodes and create new children for each one

        for (auto iter = mInputTree.root().cbeginChildOn(); iter; ++iter) {
            auto* child = new FloatTree::RootNodeType::ChildNodeType(iter.getCoord(), 0.0f, false);
            root.addChild(child);
        }

        // iterate over the source tree root tiles and create new tiles for each one

        for (auto iter = mInputTree.root().cbeginValueOn(); iter; ++iter) {
            float value = *iter > threshold ? threshold : *iter;
            root.addTile(iter.getCoord(), value, true);
        }

        return true; // continue recursing
    }

    template <typename T>
    bool operator()(T& node, size_t) const
    {
        // query the corresponding node in the source tree
        const auto* sourceNode = mInputTree.root().template probeConstNode<T>(node.origin());

        // iterate over the source child nodes

        for (auto iter = sourceNode->cbeginChildOn(); iter; ++iter) {
            if constexpr (node.LEVEL == 1) {
                // collapse any leaf nodes into tiles where the collapseLeaf() conditional is met
                if (collapseLeaf(*iter)) {
                    node.addTile(iter.pos(), threshold, true);
                    continue;
                }
            }

            auto* child = new typename T::ChildNodeType(iter.getCoord(), 0.0f, false);
            if constexpr (node.LEVEL == 1) {
                // update the values of the target leaf node to match the source leaf node
#if 1
                // this is the micro-optimized source
                const auto* sourceData = iter->buffer().data();
                auto* targetData = child->buffer().data();
                for (auto valueIter = iter->cbeginValueOn(); valueIter; ++valueIter) {
                    Index idx = valueIter.pos();
                    float value = sourceData[idx];
                    if (value > threshold)      value = threshold;
                    targetData[idx] = value;
                }
                child->setValueMask(iter->getValueMask());
#else
                // this is the vanilla implementation
                for (auto valueIter = iter->cbeginValueOn(); valueIter; ++valueIter) {
                    float value = *valueIter > threshold ? threshold : *valueIter;
                    child->addTile(valueIter.pos(), value, true);
                }
#endif
            } else {
                // iterate over the source tiles and create target tiles in the child node

                for (auto valueIter = iter->cbeginValueOn(); valueIter; ++valueIter) {
                    float value = *valueIter > threshold ? threshold : *valueIter;
                    child->addTile(valueIter.pos(), value, true);
                }
            }
            node.addChild(child);
        }

        // iterate over the source tiles and create target tiles

        for (auto iter = sourceNode->cbeginValueOn(); iter; ++iter) {
            float value = *iter > threshold ? threshold : *iter;
            node.addTile(iter.pos(), value, true);
        }

        return true; // continue recursing
    }

    bool operator()(FloatTree::LeafNodeType&, size_t) const
    {
        return true;
    }

    const FloatTree& mInputTree;
}; // struct DynamicBuildOp

void dynamicBuild(const FloatTree& inputTree)
{
    util::CpuTimer timer;
    timer.start("Dynamic Build");

    FloatTree tree(0.0f);

    // build the new float tree dynamically but skip evaluation of the leaf nodes (DEPTH-2).

    tree::DynamicNodeManager<FloatTree, FloatTree::DEPTH-2> nodeManager(tree);
    DynamicBuildOp dynamicBuildOp(inputTree);
    nodeManager.foreachTopDown(dynamicBuildOp);

    // prune any remaining internal nodes
    tools::pruneTiles(tree);

    timer.stop();
}

///////////////////////////////////////////////////////////////////////

int
main(int argc, char *argv[])
{
    ///////////////////////////////////////////////////////////////////////
    // Initialize OpenVDB and read in the Disney Cloud VDB from disk

    openvdb::initialize();

    OptParse parser(argc, argv, /*vdbArg=*/true, /*cpusArg=*/false);
    int iterations = parser.iterations();

    util::CpuTimer timer;
    timer.start("Reading Cloud VDB");
    FloatTree inputTree = openVDBAsset(parser.vdb());
    timer.stop();

    ///////////////////////////////////////////////////////////////////////
    // Prepare the cloud VDB for accurate benchmarking between techniques

    // There are a couple of assumptions made in these benchmarks which we enforce here:
    // - all inactive values have the background 0.0f value
    // - all values with the background 0.0f value are inactive
    // - all leaf nodes and internal nodes that could be pruned have been

    // set all inactive values to the background value
    for (auto iter = inputTree.beginValueOff(); iter; ++iter) {
        iter.setValue(inputTree.background());
    }
    // make all background values inactive
    for (auto iter = inputTree.beginValueOn(); iter; ++iter) {
        if (iter.getValue() == 0.0f) {
            iter.setValueOff();
        }
    }

    // prune the tree
    tools::prune(inputTree);

    ///////////////////////////////////////////////////////////////////////
    // Run the benchmarks

    // technique 1:
    // - deep copy the tree
    // - clamp the values
    // - prune the tree
    copyClampAndPrune(inputTree);

    // technique 2:
    // - copy the topology of the tree to produce a new mask tree
    // - collapse leaf nodes in this mask tree where prune would have done so
    // - create a new float tree using topology of this mask tree
    // - copy values from source tree to the resulting active values
    maskBuild(inputTree);

    // technique 3:
    // - create a thread-local storage container
    // - scatter pruned trees into each thread-local tree
    // - merge the trees
    gridReduce(inputTree);

    // technique 4:
    // - dynamically build a new float tree using the DynamicNodeManager
    dynamicBuild(inputTree);

    return 0;
}
