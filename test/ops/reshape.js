'use strict';
import * as utils from '../utils.js';

describe('test reshape', async function() {
  let device;
  let context;
  before(async () => {
    const adaptor = await navigator.gpu.requestAdapter();
    device = await adaptor.requestDevice();
    context = navigator.ml.createContext(device);
  });

  async function testReshape(oldShape, newShape, expectedShape) {
    const builder = new MLGraphBuilder(context);
    const x = builder.input('x', {type: 'float32', dimensions: oldShape});
    const y = builder.reshape(x, newShape);
    const graph = builder.build({y});
    const inputBufferSize = utils.sizeOfShape(oldShape);
    const inputArray = new Array(inputBufferSize);
    for (let i = 0; i < inputArray.length; ++i) {
      inputArray[i] = Math.random();
    }
    const inputBuffer = await utils.createGPUBuffer(device, inputBufferSize, inputArray);
    const inputs = {'x': {resource: inputBuffer}};
    const outputBufferSize = utils.sizeOfShape(expectedShape ? expectedShape : newShape);
    const outputBuffer = await utils.createGPUBuffer(device, outputBufferSize);
    const outputs = {'y': {resource: outputBuffer}};
    graph.compute(inputs, outputs);
    utils.checkValue(await utils.readbackGPUBuffer(device, outputBufferSize, outputBuffer), inputArray);
  }

  it('reshape reordered_all_dims', async function() {
    await testReshape([2, 3, 4], [4, 2, 3]);
  });

  it('reshape reordered_last_dims', async function() {
    await testReshape([2, 3, 4], [2, 4, 3]);
  });

  it('reshape reduced_dims', async function() {
    await testReshape([2, 3, 4], [2, 12]);
  });

  it('reshape extended_dims', async function() {
    await testReshape([2, 3, 4], [2, 3, 2, 2]);
  });

  it('reshape one_dim', async function() {
    await testReshape([2, 3, 4], [24]);
  });

  it('reshape [2, 3, 4] to negative_dim [2, -1, 2]', async function() {
    await testReshape([2, 3, 4], [2, -1, 2], [2, 6, 2]);
  });

  it('reshape [2, 3, 4] to negative_dim [-1, 2, 3, 4]', async function() {
    await testReshape([2, 3, 4], [-1, 2, 3, 4], [1, 2, 3, 4]);
  });
});
