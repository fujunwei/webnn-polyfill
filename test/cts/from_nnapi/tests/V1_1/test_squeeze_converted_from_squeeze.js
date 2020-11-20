'use strict';
import * as utils from '../../../../utils.js';

describe('CTS converted from NNAPI CTS', function() {
  const nn = navigator.ml.getNeuralNetworkContext();

  it('test squeeze converted from squeeze test', async function() {
    // Converted test case (from: V1_1/squeeze.mod.py)
    const builder = nn.createModelBuilder();
    const input = builder.input('input', {type: 'float32', dimensions: [4, 1, 1, 2]});
    const inputBuffer = new Float32Array([1.4, 2.3, 3.2, 4.1, 5.4, 6.3, 7.2, 8.1]);
    const squeezeDims = [1, 2];
    const expected = [1.4, 2.3, 3.2, 4.1, 5.4, 6.3, 7.2, 8.1];
    const output = builder.squeeze(input, {'axes': squeezeDims});
    const model = builder.createModel({output});
    const compilation = await model.compile();
    const outputs = await compilation.compute({'input': {buffer: inputBuffer}});
    utils.checkValue(outputs.output.buffer, expected, 1e-5, 5.0 * 1.1920928955078125e-7);
  });
});
