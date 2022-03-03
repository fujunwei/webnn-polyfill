'use strict';
import * as utils from '../utils.js';

describe('test reshape', async function() {
  const context = navigator.ml.createContext();

  function testReshape(oldShape, newShape, expectedShape) {
    const builder = new MLGraphBuilder(context);
    const x = builder.input('x', {type: 'float32', dimensions: oldShape});
    const y = builder.reshape(x, newShape);
    const graph = builder.build({y});
    const bufferSize = utils.sizeOfShape(oldShape);
    const inputBuffer = new Float32Array(bufferSize);
    for (let i = 0; i < inputBuffer.length; ++i) {
      inputBuffer[i] = Math.random();
    }
    const inputs = {'x': inputBuffer};
    const outputs = {
      'y': new Float32Array(
          utils.sizeOfShape(expectedShape ? expectedShape : newShape)),
    };
    await graph.computeAsync(inputs, outputs);
    utils.checkValue(outputs.y, inputBuffer);
  }

  it('reshape reordered_all_dims', async function() {
    testReshape([2, 3, 4], [4, 2, 3]);
  });

  it('reshape reordered_last_dims', async function() {
    testReshape([2, 3, 4], [2, 4, 3]);
  });

  it('reshape reduced_dims', async function() {
    testReshape([2, 3, 4], [2, 12]);
  });

  it('reshape extended_dims', async function() {
    testReshape([2, 3, 4], [2, 3, 2, 2]);
  });

  it('reshape one_dim', async function() {
    testReshape([2, 3, 4], [24]);
  });

  it('reshape [2, 3, 4] to negative_dim [2, -1, 2]', async function() {
    testReshape([2, 3, 4], [2, -1, 2], [2, 6, 2]);
  });

  it('reshape [2, 3, 4] to negative_dim [-1, 2, 3, 4]', async function() {
    testReshape([2, 3, 4], [-1, 2, 3, 4], [1, 2, 3, 4]);
  });
});
