/**
 * TreeVisualizer - D3.js-based tree visualization for Abyme recursive model
 *
 * Node states:
 * - generating (yellow): Node created, generation in progress
 * - complete (red): Generation done, but not final
 * - final (green): Leaf node with no continuation
 *
 * Edge types:
 * - subproblem (solid): Parent delegated to child (depth+1)
 * - continuation (dashed): Sequential flow (same depth)
 */

class TreeVisualizer {
    constructor(svgId) {
        this.svg = d3.select(`#${svgId}`);
        this.width = this.svg.node().parentElement.clientWidth;
        this.height = this.svg.node().parentElement.clientHeight;

        // Create main group for zoom/pan
        this.g = this.svg.append('g');

        // Create separate groups for edges and nodes (edges first = below nodes)
        this.edgeGroup = this.g.append('g').attr('class', 'edges-container');
        this.nodeGroup = this.g.append('g').attr('class', 'nodes-container');

        // Enable zoom and pan
        const zoom = d3.zoom()
            .scaleExtent([0.1, 3])
            .on('zoom', (event) => {
                this.g.attr('transform', event.transform);
            });

        this.svg.call(zoom);

        // Data structures
        this.nodes = new Map();  // node_id → {id, prompt, context, output, depth, state, latency}
        this.edges = [];         // [{source, target, type}]
        this.previousPositions = new Map();  // Track previous positions for orphaned nodes

        // Layout settings
        this.nodeRadius = 25;
        this.levelHeight = 120;
        this.nodeSpacing = 100;

        // Track stats
        this.callCount = 0;
        this.maxDepthSeen = 0;
    }

    /**
     * Reset the visualization
     */
    reset() {
        this.nodes.clear();
        this.edges = [];
        this.previousPositions.clear();
        this.callCount = 0;
        this.maxDepthSeen = 0;
        this.updateVisualization();
        this.hideEmptyState(false);
    }

    /**
     * Hide/show empty state message
     */
    hideEmptyState(hide) {
        const emptyState = document.getElementById('empty-state');
        if (emptyState) {
            emptyState.style.display = hide ? 'none' : 'flex';
        }
    }

    /**
     * Add a new node with 'generating' state (yellow)
     */
    addNode(nodeData) {
        this.hideEmptyState(true);

        const node = {
            id: nodeData.node_id,
            prompt: nodeData.prompt,
            context: nodeData.context,
            output: null,
            depth: nodeData.depth,
            state: 'generating',  // yellow
            latency: null,
            parentId: nodeData.parent_id  // Store parent relationship
        };

        this.nodes.set(node.id, node);
        this.callCount++;
        this.maxDepthSeen = Math.max(this.maxDepthSeen, node.depth);

        // If this node has a parent, create the edge immediately
        // This prevents orphaned nodes during visualization
        if (nodeData.parent_id) {
            this.addEdge(nodeData.parent_id, nodeData.node_id, 'subproblem');
        }

        // Always update visualization when adding a new node
        this.updateVisualization();
        this.updateStats();
    }

    /**
     * Update node to 'waiting' state (red-orange) - waiting for child responses
     */
    setNodeWaiting(nodeId, output, numSubproblems) {
        const node = this.nodes.get(nodeId);
        if (node) {
            node.output = output;
            node.state = 'waiting';  // red-orange
            node.numSubproblems = numSubproblems;
            this.updateVisualization();
        }
    }

    /**
     * Update node to 'complete' state (blue) with output
     */
    completeNode(nodeId, output, latency) {
        const node = this.nodes.get(nodeId);
        if (node) {
            node.output = output;
            node.latency = latency;
            node.state = 'complete';  // blue
            this.updateVisualization();
        }
    }

    /**
     * Update node to 'final' state (green)
     */
    finalizeNode(nodeId) {
        const node = this.nodes.get(nodeId);
        if (node) {
            node.state = 'final';  // green
            this.updateVisualization();
        }
    }

    /**
     * Add an edge between two nodes
     */
    addEdge(sourceId, targetId, type) {
        // Check if edge already exists to avoid duplicates
        const edgeExists = this.edges.some(edge =>
            edge.source === sourceId &&
            edge.target === targetId &&
            edge.type === type
        );

        if (!edgeExists) {
            this.edges.push({
                source: sourceId,
                target: targetId,
                type: type  // 'subproblem' or 'continuation'
            });
            this.updateVisualization();
        }
    }

    /**
     * Calculate node positions using improved tree layout
     * - First child directly below parent
     * - Other children to the right
     * - Continuation (next) to the right
     */
    calculatePositions() {
        const positions = new Map();
        const subtreeWidths = new Map();

        // Build tree structure
        const nodeMap = new Map(this.nodes);
        const children = new Map();
        const nextMap = new Map();

        // Map edges to parent-child and next relationships
        for (const edge of this.edges) {
            if (edge.type === 'subproblem') {
                if (!children.has(edge.source)) {
                    children.set(edge.source, []);
                }
                children.get(edge.source).push(edge.target);
            } else if (edge.type === 'continuation') {
                nextMap.set(edge.source, edge.target);
            }
        }

        // Find root node (depth 0)
        let root = null;
        for (const [id, node] of this.nodes) {
            if (node.depth === 0) {
                root = id;
                break;
            }
        }

        if (!root) return positions;

        // Calculate subtree width for each node (bottom-up)
        const calculateSubtreeWidth = (nodeId) => {
            if (subtreeWidths.has(nodeId)) {
                return subtreeWidths.get(nodeId);
            }

            const nodeChildren = children.get(nodeId) || [];
            const nextId = nextMap.get(nodeId);

            // Calculate width from children
            let totalWidth = 0;
            if (nodeChildren.length > 0) {
                for (const childId of nodeChildren) {
                    totalWidth += calculateSubtreeWidth(childId);
                }
                // Add extra spacing between siblings
                totalWidth += (nodeChildren.length - 1) * this.nodeSpacing * 0.5;
            } else {
                // No children, so this node itself takes spacing
                totalWidth = this.nodeSpacing;
            }

            // Add continuation width if exists (continuation is on same level, to the right)
            if (nextId) {
                const nextWidth = calculateSubtreeWidth(nextId);
                totalWidth += nextWidth + this.nodeSpacing * 1.5;
            }

            subtreeWidths.set(nodeId, totalWidth);
            return totalWidth;
        };

        // Position nodes (top-down) and return rightmost X position
        const positionNode = (nodeId, x, y) => {
            positions.set(nodeId, { x, y });

            const nodeChildren = children.get(nodeId) || [];
            const nodeWidth = subtreeWidths.get(nodeId) || this.nodeSpacing;

            // Start with this node's right extent (excluding continuation)
            const nextId = nextMap.get(nodeId);
            let rightmostX = x;

            if (nodeChildren.length > 0) {
                // First child directly below
                const firstChild = nodeChildren[0];
                positionNode(firstChild, x, y + this.levelHeight);
                const firstChildWidth = subtreeWidths.get(firstChild) || this.nodeSpacing;
                rightmostX = x + firstChildWidth;

                // Other children to the right
                let currentX = x + firstChildWidth;
                for (let i = 1; i < nodeChildren.length; i++) {
                    const childId = nodeChildren[i];
                    currentX += this.nodeSpacing * 0.5;
                    positionNode(childId, currentX, y + this.levelHeight);
                    const childWidth = subtreeWidths.get(childId) || this.nodeSpacing;
                    currentX += childWidth;
                    rightmostX = currentX;
                }
            } else {
                // Leaf node - rightmost is x + nodeSpacing
                rightmostX = x + this.nodeSpacing;
            }

            // Position continuation (next) to the right of entire subtree
            if (nextId) {
                const nextX = rightmostX + this.nodeSpacing * 1.5;
                positionNode(nextId, nextX, y);
                const nextWidth = subtreeWidths.get(nextId) || this.nodeSpacing;
                rightmostX = nextX + nextWidth;
            }

            return rightmostX;
        };

        // Calculate all subtree widths
        calculateSubtreeWidth(root);

        // Position all nodes starting from root
        positionNode(root, 100, 60);

        // Handle any remaining orphaned nodes (shouldn't happen normally)
        // These would only occur for root nodes or during initialization
        for (const [nodeId, node] of this.nodes) {
            if (!positions.has(nodeId)) {
                // Root nodes or orphaned nodes get positioned at a sensible default
                const defaultX = (this.width && this.width > 0) ? this.width / 3 : 300;
                positions.set(nodeId, {
                    x: defaultX,
                    y: 60 + node.depth * this.levelHeight
                });
            }
        }

        // Save current positions for next update
        this.previousPositions = new Map(positions);

        return positions;
    }

    /**
     * Update the visualization with current nodes and edges
     */
    updateVisualization() {
        const positions = this.calculatePositions();

        // Filter edges to only include those where both nodes exist and have positions
        const validEdges = this.edges.filter(edge => {
            const hasSource = this.nodes.has(edge.source) && positions.has(edge.source);
            const hasTarget = this.nodes.has(edge.target) && positions.has(edge.target);

            if (!hasSource || !hasTarget) {
                console.warn(`Skipping edge ${edge.source}->${edge.target}:`,
                    `source exists: ${this.nodes.has(edge.source)}, positioned: ${positions.has(edge.source)}`,
                    `target exists: ${this.nodes.has(edge.target)}, positioned: ${positions.has(edge.target)}`);
                return false;
            }
            return true;
        });

        // Render edges (below nodes in z-order) - use edgeGroup container
        const edgeSelection = this.edgeGroup.selectAll('.edge-group')
            .data(validEdges, d => `${d.source}-${d.target}`);

        // Enter new edge groups
        const edgeEnter = edgeSelection.enter()
            .append('g')
            .attr('class', 'edge-group');

        // Add line to edge group
        edgeEnter.append('line')
            .attr('class', 'edge-line');

        // Add arrowhead for continuation edges
        edgeEnter.append('path')
            .attr('class', 'edge-arrow');

        // Update edge groups
        const edgeUpdate = edgeEnter.merge(edgeSelection);

        // Update lines with smooth transitions
        edgeUpdate.select('.edge-line')
            .transition()
            .duration(300)
            .attr('x1', d => {
                const pos = positions.get(d.source);
                return pos && typeof pos.x === 'number' ? pos.x : 0;
            })
            .attr('y1', d => {
                const pos = positions.get(d.source);
                return pos && typeof pos.y === 'number' ? pos.y : 0;
            })
            .attr('x2', d => {
                const pos = positions.get(d.target);
                if (!pos || typeof pos.x !== 'number') return 0;
                // For continuation, shorten line to make room for arrow
                if (d.type === 'continuation') {
                    const sourcePos = positions.get(d.source);
                    if (!sourcePos || typeof sourcePos.x !== 'number' || typeof sourcePos.y !== 'number') return pos.x;
                    const dx = pos.x - sourcePos.x;
                    const dy = pos.y - sourcePos.y;
                    const len = Math.sqrt(dx * dx + dy * dy);
                    if (len === 0) return pos.x;
                    const ratio = (len - this.nodeRadius - 10) / len;
                    return sourcePos.x + dx * ratio;
                }
                return pos.x;
            })
            .attr('y2', d => {
                const pos = positions.get(d.target);
                if (!pos || typeof pos.y !== 'number') return 0;
                // For continuation, shorten line to make room for arrow
                if (d.type === 'continuation') {
                    const sourcePos = positions.get(d.source);
                    if (!sourcePos || typeof sourcePos.x !== 'number' || typeof sourcePos.y !== 'number') return pos.y;
                    const dx = pos.x - sourcePos.x;
                    const dy = pos.y - sourcePos.y;
                    const len = Math.sqrt(dx * dx + dy * dy);
                    if (len === 0) return pos.y;
                    const ratio = (len - this.nodeRadius - 10) / len;
                    return sourcePos.y + dy * ratio;
                }
                return pos.y;
            })
            .attr('stroke', '#FFFFFF')
            .attr('stroke-width', d => d.type === 'continuation' ? 4 : 3)
            .attr('stroke-dasharray', d => d.type === 'continuation' ? '0' : '0')
            .attr('opacity', 0.9)
            .style('filter', 'drop-shadow(0 0 4px rgba(255, 255, 255, 0.6)) drop-shadow(0 0 8px rgba(255, 255, 255, 0.4))');

        // Update arrows (only for continuation edges) with smooth transitions
        edgeUpdate.select('.edge-arrow')
            .transition()
            .duration(300)
            .attr('d', d => {
                if (d.type !== 'continuation') return '';

                const sourcePos = positions.get(d.source);
                const targetPos = positions.get(d.target);
                if (!sourcePos || !targetPos) return '';

                // Calculate arrow at target end
                const dx = targetPos.x - sourcePos.x;
                const dy = targetPos.y - sourcePos.y;
                const len = Math.sqrt(dx * dx + dy * dy);
                const ratio = (len - this.nodeRadius - 2) / len;

                const arrowX = sourcePos.x + dx * ratio;
                const arrowY = sourcePos.y + dy * ratio;

                // Arrow direction
                const angle = Math.atan2(dy, dx);
                const arrowSize = 12;

                // Calculate arrow points
                const p1x = arrowX;
                const p1y = arrowY;
                const p2x = arrowX - arrowSize * Math.cos(angle - Math.PI / 6);
                const p2y = arrowY - arrowSize * Math.sin(angle - Math.PI / 6);
                const p3x = arrowX - arrowSize * Math.cos(angle + Math.PI / 6);
                const p3y = arrowY - arrowSize * Math.sin(angle + Math.PI / 6);

                return `M ${p1x},${p1y} L ${p2x},${p2y} L ${p3x},${p3y} Z`;
            })
            .attr('fill', '#FFFFFF')
            .attr('opacity', 0.9)
            .style('filter', 'drop-shadow(0 0 4px rgba(255, 255, 255, 0.6)) drop-shadow(0 0 8px rgba(255, 255, 255, 0.4))');

        // Remove old edges
        edgeSelection.exit().remove();

        // Render nodes - use nodeGroup container
        const nodeData = Array.from(this.nodes.values());
        const nodeSelection = this.nodeGroup.selectAll('.node')
            .data(nodeData, d => d.id);

        // Enter new nodes
        const nodeEnter = nodeSelection.enter()
            .append('g')
            .attr('class', 'node')
            .attr('transform', d => {
                const pos = positions.get(d.id);
                if (!pos) {
                    console.warn('No position found for node:', d.id);
                    return `translate(0, 0)`;
                }
                return `translate(${pos.x}, ${pos.y})`;
            });

        // Add circle to each node
        nodeEnter.append('circle')
            .attr('r', this.nodeRadius)
            .attr('stroke', '#fff')
            .attr('stroke-width', 2.5);

        // Add pulsing animation for generating nodes
        nodeEnter.append('circle')
            .attr('class', 'pulse-ring')
            .attr('r', this.nodeRadius)
            .attr('fill', 'none')
            .attr('stroke-width', 0);

        // Merge and update
        const nodeUpdate = nodeEnter.merge(nodeSelection);

        // Animate position changes smoothly
        nodeUpdate
            .transition()
            .duration(300)
            .attr('transform', d => {
                const pos = positions.get(d.id);
                if (!pos) {
                    console.warn('No position found for node during update:', d.id);
                    return `translate(0, 0)`;
                }
                return `translate(${pos.x}, ${pos.y})`;
            });

        // Update circle colors
        nodeUpdate.select('circle:first-child')
            .attr('fill', d => this.getNodeColor(d.state))
            .style('cursor', 'pointer');

        // Update pulse animation
        nodeUpdate.select('.pulse-ring')
            .attr('stroke', d => d.state === 'generating' ? this.getNodeColor(d.state) : 'none')
            .attr('stroke-width', d => d.state === 'generating' ? 3 : 0)
            .style('animation', d => d.state === 'generating' ? 'pulse 1.5s ease-out infinite' : 'none');

        // Add hover and click events
        nodeUpdate
            .on('mouseenter', (event, d) => this.showTooltip(event, d))
            .on('mouseleave', () => this.hideTooltip())
            .on('mousemove', (event) => this.moveTooltip(event))
            .on('click', (event, d) => {
                event.stopPropagation();
                this.showSidebar(d);
            });

        // Remove old nodes
        nodeSelection.exit().remove();

        // Center view on first render if there's a root node
        if (this.nodes.size === 1) {
            const rootPos = positions.values().next().value;
            if (rootPos) {
                const transform = d3.zoomIdentity
                    .translate(this.width / 2 - rootPos.x, 50)
                    .scale(1);
                this.svg.transition().duration(750).call(
                    d3.zoom().transform,
                    transform
                );
            }
        }
    }

    /**
     * Get color based on node state
     */
    getNodeColor(state) {
        switch(state) {
            case 'generating': return '#EAB308';  // Yellow
            case 'waiting': return '#F97316';     // Red-orange (waiting for children)
            case 'complete': return '#3B82F6';    // Blue
            case 'final': return '#22C55E';       // Green
            default: return '#6B7280';            // Gray
        }
    }

    /**
     * Show tooltip with node details
     */
    showTooltip(event, nodeData) {
        const tooltip = document.getElementById('tooltip');
        const content = document.getElementById('tooltip-content');

        if (!tooltip || !content) return;

        let html = `
            <div class="text-gray-300">
                <span class="text-gray-400 font-semibold">Depth:</span> ${nodeData.depth}
            </div>
        `;

        // Prompt (truncate if too long)
        const promptText = nodeData.prompt.length > 200
            ? nodeData.prompt.substring(0, 200) + '...'
            : nodeData.prompt;

        html += `
            <div class="text-gray-300">
                <span class="text-gray-400 font-semibold">Prompt:</span><br/>
                <span class="font-mono text-xs bg-gray-900 p-2 block rounded mt-1">${this.escapeHtml(promptText)}</span>
            </div>
        `;

        // Context (if exists)
        if (nodeData.context) {
            const contextText = nodeData.context.length > 200
                ? nodeData.context.substring(0, 200) + '...'
                : nodeData.context;
            html += `
                <div class="text-gray-300">
                    <span class="text-gray-400 font-semibold">Context:</span><br/>
                    <span class="font-mono text-xs bg-gray-900 p-2 block rounded mt-1">${this.escapeHtml(contextText)}</span>
                </div>
            `;
        }

        // Output (only if node is complete or final)
        if (nodeData.output && nodeData.state !== 'generating') {
            const outputText = nodeData.output.length > 200
                ? nodeData.output.substring(0, 200) + '...'
                : nodeData.output;
            html += `
                <div class="text-gray-300">
                    <span class="text-gray-400 font-semibold">Output:</span><br/>
                    <span class="font-mono text-xs bg-gray-900 p-2 block rounded mt-1">${this.escapeHtml(outputText)}</span>
                </div>
            `;
        }

        // Latency (if available)
        if (nodeData.latency !== null) {
            html += `
                <div class="text-gray-300">
                    <span class="text-gray-400 font-semibold">Latency:</span> ${nodeData.latency.toFixed(2)}s
                </div>
            `;
        }

        // State
        html += `
            <div class="text-gray-300 mt-2 pt-2 border-t border-gray-700">
                <span class="text-gray-400 font-semibold">State:</span>
                <span class="inline-block px-2 py-0.5 rounded text-xs" style="background-color: ${this.getNodeColor(nodeData.state)}20; color: ${this.getNodeColor(nodeData.state)}">
                    ${nodeData.state}
                </span>
                ${nodeData.state === 'waiting' && nodeData.numSubproblems ? `
                    <span class="text-gray-400 text-xs ml-2">(waiting for ${nodeData.numSubproblems} subproblems)</span>
                ` : ''}
            </div>
        `;

        content.innerHTML = html;

        // Position tooltip
        this.moveTooltip(event);
        tooltip.classList.remove('hidden');
    }

    /**
     * Move tooltip to follow mouse
     */
    moveTooltip(event) {
        const tooltip = document.getElementById('tooltip');
        if (!tooltip) return;

        const offsetX = 15;
        const offsetY = 15;

        tooltip.style.left = (event.pageX + offsetX) + 'px';
        tooltip.style.top = (event.pageY + offsetY) + 'px';
    }

    /**
     * Hide tooltip
     */
    hideTooltip() {
        const tooltip = document.getElementById('tooltip');
        if (tooltip) {
            tooltip.classList.add('hidden');
        }
    }

    /**
     * Update stats display
     */
    updateStats() {
        const callCountElem = document.getElementById('call-count');
        const currentDepthElem = document.getElementById('current-depth');

        if (callCountElem) {
            callCountElem.textContent = this.callCount;
        }

        if (currentDepthElem) {
            currentDepthElem.textContent = this.maxDepthSeen;
        }
    }

    /**
     * Escape HTML to prevent XSS
     */
    escapeHtml(text) {
        const map = {
            '&': '&amp;',
            '<': '&lt;',
            '>': '&gt;',
            '"': '&quot;',
            "'": '&#039;'
        };
        return text.replace(/[&<>"']/g, m => map[m]);
    }

    /**
     * Show sidebar with full node details
     */
    showSidebar(nodeData) {
        const sidebar = document.getElementById('node-sidebar');
        const content = document.getElementById('sidebar-content');

        if (!sidebar || !content) return;

        // Render markdown-like formatting
        const renderText = (text) => {
            if (!text) return '<span class="text-gray-500 italic">None</span>';

            // Escape HTML first
            let html = this.escapeHtml(text);

            // Code blocks (with optional language) - preserve these first
            const codeBlocks = [];
            html = html.replace(/```(\w+)?\s*\n([\s\S]*?)```/g, (match, lang, code) => {
                const placeholder = `___CODEBLOCK_${codeBlocks.length}___`;
                codeBlocks.push(`<pre class="bg-gray-900 p-3 rounded overflow-x-auto my-2 border border-gray-700"><code class="text-sm text-green-400">${code.trim()}</code></pre>`);
                return placeholder;
            });

            // Inline code - preserve these too
            const inlineCodes = [];
            html = html.replace(/`([^`]+)`/g, (match, code) => {
                const placeholder = `___INLINECODE_${inlineCodes.length}___`;
                inlineCodes.push(`<code class="bg-gray-900 px-1.5 py-0.5 rounded text-sm text-green-400 border border-gray-700">${code}</code>`);
                return placeholder;
            });

            // Headings (process in order from most specific to least)
            html = html.replace(/^### (.+)$/gm, '<h3 class="text-lg font-bold text-blue-300 mt-3 mb-2">$1</h3>');
            html = html.replace(/^## (.+)$/gm, '<h2 class="text-xl font-bold text-blue-300 mt-4 mb-2">$1</h2>');
            html = html.replace(/^# (.+)$/gm, '<h1 class="text-2xl font-bold text-blue-300 mt-4 mb-3">$1</h1>');

            // Bold (must come before italic to handle ** before *)
            html = html.replace(/\*\*(.+?)\*\*/g, '<strong class="font-bold text-white">$1</strong>');

            // Italic
            html = html.replace(/\*(.+?)\*/g, '<em class="italic text-gray-300">$1</em>');

            // Links
            html = html.replace(/\[([^\]]+)\]\(([^)]+)\)/g, '<a href="$2" class="text-blue-400 hover:text-blue-300 underline" target="_blank">$1</a>');

            // Horizontal rules
            html = html.replace(/^---+$/gm, '<hr class="border-gray-700 my-3">');

            // Blockquotes
            html = html.replace(/^&gt; (.+)$/gm, '<blockquote class="border-l-4 border-blue-500 pl-3 italic text-gray-400 my-2">$1</blockquote>');

            // Unordered lists
            html = html.replace(/^- (.+)$/gm, '<li class="ml-4">• $1</li>');
            html = html.replace(/(<li class="ml-4">• .*?<\/li>\n?)+/g, '<ul class="my-2 space-y-1">$&</ul>');

            // Ordered lists
            html = html.replace(/^\d+\. (.+)$/gm, '<li class="ml-4 list-decimal">$1</li>');
            html = html.replace(/(<li class="ml-4 list-decimal">.*?<\/li>\n?)+/g, '<ol class="my-2 space-y-1 list-decimal ml-8">$&</ol>');

            // Line breaks (do this LAST, after all pattern matching)
            html = html.replace(/\n\n/g, '<br><br>');
            html = html.replace(/\n/g, '<br>');

            // Restore code blocks and inline code
            codeBlocks.forEach((block, i) => {
                html = html.replace(`___CODEBLOCK_${i}___`, block);
            });
            inlineCodes.forEach((code, i) => {
                html = html.replace(`___INLINECODE_${i}___`, code);
            });

            return html;
        };

        const html = `
            <div class="space-y-4">
                <div>
                    <div class="text-sm font-semibold text-gray-400 mb-1">Status</div>
                    <div>
                        <span class="inline-block px-3 py-1 rounded text-sm font-medium"
                              style="background-color: ${this.getNodeColor(nodeData.state)}20; color: ${this.getNodeColor(nodeData.state)}">
                            ${nodeData.state}
                        </span>
                        ${nodeData.state === 'waiting' && nodeData.numSubproblems ? `
                            <div class="text-xs text-gray-400 mt-1">Waiting for ${nodeData.numSubproblems} subproblems</div>
                        ` : ''}
                    </div>
                </div>

                <div>
                    <div class="text-sm font-semibold text-gray-400 mb-1">Depth</div>
                    <div class="text-white">${nodeData.depth}</div>
                </div>

                ${nodeData.latency !== null ? `
                    <div>
                        <div class="text-sm font-semibold text-gray-400 mb-1">Latency</div>
                        <div class="text-white">${nodeData.latency.toFixed(2)}s</div>
                    </div>
                ` : ''}

                <div>
                    <div class="text-sm font-semibold text-gray-400 mb-2">Prompt</div>
                    <div class="bg-gray-800 p-3 rounded max-h-64 overflow-y-auto text-sm text-gray-100 markdown-content">
                        ${renderText(nodeData.prompt)}
                    </div>
                </div>

                ${nodeData.context ? `
                    <div>
                        <div class="text-sm font-semibold text-gray-400 mb-2">Context</div>
                        <div class="bg-gray-800 p-3 rounded max-h-64 overflow-y-auto text-sm text-gray-100 markdown-content">
                            ${renderText(nodeData.context)}
                        </div>
                    </div>
                ` : ''}

                ${nodeData.output && nodeData.state !== 'generating' ? `
                    <div>
                        <div class="text-sm font-semibold text-gray-400 mb-2">Output</div>
                        <div class="bg-gray-800 p-3 rounded max-h-96 overflow-y-auto text-sm text-gray-100 markdown-content">
                            ${renderText(nodeData.output)}
                        </div>
                    </div>
                ` : ''}
            </div>
        `;

        content.innerHTML = html;
        sidebar.classList.remove('hidden');
        sidebar.classList.add('flex');
    }

    /**
     * Hide sidebar
     */
    hideSidebar() {
        const sidebar = document.getElementById('node-sidebar');
        if (sidebar) {
            sidebar.classList.add('hidden');
            sidebar.classList.remove('flex');
        }
    }
}
