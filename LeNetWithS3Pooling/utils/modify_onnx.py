import torch
import onnx
import argparse

def main():
    # Configurable parameters from command line
    parser = argparse.ArgumentParser(description='ONNX Modifying Example')
    parser.add_argument('--onnx',
                        help='onnx file to modify')
    parser.add_argument('--output', default="output.onnx",
                        help='input batch size for testing (default: output.onnx)')
    args = parser.parse_args()

    # Load ONNX file
    model = onnx.load(args.onnx)

    # Retrieve graph_def
    graph = model.graph
    
    # List of nodes in the graph
    nodes = graph.node
    prev_node = None

    # Iterate through all the nodes
    for node in nodes:
        # Search for Pad + MaxPool layer for modification in ONNX export
        if prev_node and prev_node.op_type == "Pad" and node.op_type == "MaxPool":
            # Modify the Padding layer as per Average Pooling
            dup_prev_node = onnx.helper.make_node("Pad",
                            inputs=[x for x in prev_node.input],
                            outputs=[x for x in prev_node.output],
                            mode='constant',
                            value=0.0,
                            pads=[0, 0, 0, 0, 0, 0, 0, 0],
                            )
            # Replace the Padding node with new padding node
            graph.node.remove(prev_node)
            graph.node.extend([dup_prev_node])

            # Modify the pooling with S3Pooling for Webinar demo
            node.op_type ="S3Pool"
        prev_node = node

    # Generate model_definition from modified graph
    model_def = onnx.helper.make_model(graph)

    # Save the serialized model
    onnx.save(model_def, args.output)
        
if __name__ == '__main__':
    main()
