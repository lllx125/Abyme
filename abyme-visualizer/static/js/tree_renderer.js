/**
 * Tree Renderer Module
 * D3.js-based tree visualization with live updates
 */

import { logger } from './logger.js';

const MODULE = 'TreeRenderer';

// Node status colors (matching NodeStatus in tree_trace.py)
const NODE_COLORS = {
    'WAIT_GEN': '#808080',      // grey
    'GENERATING': '#FFD700',    // yellow (with pulsing animation)
    'WAIT_SUB': '#FF6347',      // orange-red
    'COMPLETED': '#4169E1',     // blue
    'FINAL': '#32CD32',         // green
    'FAILED': '#FF0000',        // red
    'CANCELLED': '#404040'      // dark grey
};

// Edge type styles
const EDGE_STYLES = {
    'AND': { stroke: '#FF69B4', strokeWidth: 2, class: 'edge-and' },  // pink, glowing
    'OR': { stroke: '#00FF00', strokeWidth: 2, class: 'edge-or' },    // lime, glowing
    'PAST': { stroke: '#FFFFFF', strokeWidth: 1.5, class: 'edge-past' }  // white arrow, glowing
};

export class TreeRenderer {
    constructor(canvasId, onNodeClick) {
        this.canvasId = canvasId;
        this.onNodeClick = onNodeClick;
        this.svg = null;
        this.g = null;
        this.zoom = null;
        this.nodes = [];
        this.edges = [];
        this.tooltip = null;

        this.init();
    }

    /**
     * Initialize the SVG canvas and D3 elements
     */
    init() {
        logger.info(MODULE, 'Initializing tree renderer');

        // Get container dimensions
        const container = d3.select(`#${this.canvasId}`).node().parentElement;
        const width = container.clientWidth;
        const height = container.clientHeight;

        // Create SVG
        this.svg = d3.select(`#${this.canvasId}`)
            .attr('width', width)
            .attr('height', height);

        // Create zoom behavior
        this.zoom = d3.zoom()
            .scaleExtent([0.1, 4])
            .on('zoom', (event) => {
                this.g.attr('transform', event.transform);
            });

        this.svg.call(this.zoom);

        // Create main group for all elements
        this.g = this.svg.append('g');

        // Define arrow markers for PAST edges
        this.svg.append('defs').append('marker')
            .attr('id', 'arrow')
            .attr('viewBox', '0 -5 10 10')
            .attr('refX', 8)
            .attr('refY', 0)
            .attr('markerWidth', 6)
            .attr('markerHeight', 6)
            .attr('orient', 'auto')
            .append('path')
            .attr('d', 'M0,-5L10,0L0,5')
            .attr('fill', '#FFFFFF');

        // Define glow filter for edges
        const filter = this.svg.select('defs').append('filter')
            .attr('id', 'glow');

        filter.append('feGaussianBlur')
            .attr('stdDeviation', '3')
            .attr('result', 'coloredBlur');

        const feMerge = filter.append('feMerge');
        feMerge.append('feMergeNode').attr('in', 'coloredBlur');
        feMerge.append('feMergeNode').attr('in', 'SourceGraphic');

        // Create tooltip
        this.tooltip = d3.select('body').append('div')
            .attr('class', 'tree-tooltip')
            .style('opacity', 0)
            .style('position', 'absolute')
            .style('background', 'rgba(0, 0, 0, 0.8)')
            .style('color', 'white')
            .style('padding', '10px')
            .style('border-radius', '5px')
            .style('pointer-events', 'none')
            .style('font-size', '12px')
            .style('max-width', '300px')
            .style('z-index', '1000');

        // Center initial view
        this.centerView(width, height);

        logger.info(MODULE, 'Tree renderer initialized', { width, height });
    }

    /**
     * Center the view
     */
    centerView(width, height) {
        const initialTransform = d3.zoomIdentity
            .translate(width / 2, height / 2)
            .scale(1);
        this.svg.call(this.zoom.transform, initialTransform);
    }

    /**
     * Update the tree with new data
     * @param {array} nodes - Array of node objects
     * @param {array} edges - Array of edge objects
     */
    updateTree(nodes, edges) {
        logger.info(MODULE, 'Updating tree', { nodeCount: nodes.length, edgeCount: edges.length });

        this.nodes = nodes;
        this.edges = edges;

        this.render();
    }

    /**
     * Render the tree
     */
    render() {
        // Render edges first (so they appear behind nodes)
        this.renderEdges();

        // Then render nodes
        this.renderNodes();
    }

    /**
     * Render edges
     */
    renderEdges() {
        const edgeSelection = this.g.selectAll('.edge')
            .data(this.edges, (d, i) => `edge-${i}`);

        // Remove old edges
        edgeSelection.exit()
            .transition()
            .duration(300)
            .style('opacity', 0)
            .remove();

        // Add new edges
        const edgeEnter = edgeSelection.enter()
            .append('line')
            .attr('class', d => `edge ${EDGE_STYLES[d.type]?.class || 'edge-default'}`)
            .attr('x1', d => d.from.x)
            .attr('y1', d => d.from.y)
            .attr('x2', d => d.to.x)
            .attr('y2', d => d.to.y)
            .attr('stroke', d => EDGE_STYLES[d.type]?.stroke || '#888')
            .attr('stroke-width', d => EDGE_STYLES[d.type]?.strokeWidth || 1)
            .attr('marker-end', d => d.type === 'PAST' ? 'url(#arrow)' : null)
            .style('opacity', 0);

        // Update existing + new edges
        edgeSelection.merge(edgeEnter)
            .transition()
            .duration(300)
            .attr('x1', d => d.from.x)
            .attr('y1', d => d.from.y)
            .attr('x2', d => d.to.x)
            .attr('y2', d => d.to.y)
            .style('opacity', 1);
    }

    /**
     * Render nodes
     */
    renderNodes() {
        const nodeSelection = this.g.selectAll('.node-group')
            .data(this.nodes, d => `${d.prompt}-${d.x}-${d.y}`);

        // Remove old nodes
        nodeSelection.exit()
            .transition()
            .duration(300)
            .attr('transform', d => `translate(${d.x},${d.y}) scale(0)`)
            .style('opacity', 0)
            .remove();

        // Add new nodes
        const nodeEnter = nodeSelection.enter()
            .append('g')
            .attr('class', 'node-group')
            .attr('transform', d => `translate(${d.x},${d.y}) scale(0)`)
            .style('cursor', 'pointer')
            .on('click', (event, d) => {
                event.stopPropagation();
                logger.info(MODULE, 'Node clicked', d);
                if (this.onNodeClick) {
                    this.onNodeClick(d);
                }
            })
            .on('mouseenter', (event, d) => this.showTooltip(event, d))
            .on('mousemove', (event) => this.moveTooltip(event))
            .on('mouseleave', () => this.hideTooltip());

        // Add circle to each node
        nodeEnter.append('circle')
            .attr('r', 20)
            .attr('fill', d => NODE_COLORS[d.status] || '#888')
            .attr('stroke', '#fff')
            .attr('stroke-width', 2)
            .attr('class', d => d.status === 'GENERATING' ? 'pulsing' : '');

        // Add text label (node type)
        nodeEnter.append('text')
            .attr('text-anchor', 'middle')
            .attr('dy', '0.3em')
            .attr('fill', '#000')
            .attr('font-size', '10px')
            .attr('font-weight', 'bold')
            .text(d => d.type || '');

        // Update existing + new nodes
        const nodeUpdate = nodeSelection.merge(nodeEnter);

        nodeUpdate.transition()
            .duration(300)
            .attr('transform', d => `translate(${d.x},${d.y}) scale(1)`)
            .style('opacity', 1);

        // Update circle colors based on status
        nodeUpdate.select('circle')
            .transition()
            .duration(300)
            .attr('fill', d => NODE_COLORS[d.status] || '#888')
            .attr('class', d => d.status === 'GENERATING' ? 'pulsing' : '');

        // Update text
        nodeUpdate.select('text')
            .text(d => d.type || '');
    }

    /**
     * Show tooltip on hover
     */
    showTooltip(event, d) {
        const promptPreview = d.prompt.length > 100 ? d.prompt.substring(0, 100) + '...' : d.prompt;
        const outputPreview = d.output.length > 100 ? d.output.substring(0, 100) + '...' : d.output;

        const content = `
            <strong>Status:</strong> ${d.status}<br>
            <strong>Type:</strong> ${d.type}<br>
            <strong>Prompt:</strong> ${promptPreview}<br>
            ${d.output ? `<strong>Output:</strong> ${outputPreview}` : ''}
        `;

        this.tooltip
            .html(content)
            .style('opacity', 0.9);

        this.moveTooltip(event);
    }

    /**
     * Move tooltip with mouse
     */
    moveTooltip(event) {
        this.tooltip
            .style('left', (event.pageX + 15) + 'px')
            .style('top', (event.pageY - 28) + 'px');
    }

    /**
     * Hide tooltip
     */
    hideTooltip() {
        this.tooltip
            .transition()
            .duration(200)
            .style('opacity', 0);
    }

    /**
     * Clear the tree
     */
    clear() {
        this.g.selectAll('*').remove();
        this.nodes = [];
        this.edges = [];
        logger.info(MODULE, 'Tree cleared');
    }
}
