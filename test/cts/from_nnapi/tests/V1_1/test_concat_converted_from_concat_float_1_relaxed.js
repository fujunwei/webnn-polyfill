'use strict';
import * as utils from '../../../../utils.js';

/* eslint-disable max-len */
describe('CTS converted from NNAPI CTS', function() {
  const context = navigator.ml.createContext();

  it('test concat converted from concat_float_1_relaxed test', async function() {
    // Converted test case (from: V1_1/concat_float_1_relaxed.mod.py)
    const builder = new MLGraphBuilder(context);
    const op1 = builder.input('op1', {type: 'float32', dimensions: [2, 3]});
    const op1Data = new Float32Array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    const op2 = builder.input('op2', {type: 'float32', dimensions: [2, 3]});
    const op2Data = new Float32Array([7.0, 8.0, 9.0, 10.0, 11.0, 12.0]);
    const axis0 = 0;
    const expected = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0];
    const result = builder.concat([op1, op2], axis0);
    const graph = await builder.build({result});
    const outputs = await graph.compute({'op1': {data: op1Data}, 'op2': {data: op2Data}});
    utils.checkValue(outputs.result.data, expected, utils.ctsFp32RelaxedAccuracyCriteria);
  });
});
/* eslint-disable max-len */